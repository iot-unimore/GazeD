"""
Creative Commons Attribution-NonCommercial ShareAlike 4.0 International License  https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import os
import pickle
import random

import cv2
import numpy as np
import torch
from torchvision import transforms



def crop_around_person(frame, box):
    center = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))
    scale = ((box[2] - box[0]), (box[3] - box[1]))

    h = scale[1]*1
    w = 192/256*h

    xmin = int(center[0] - w/2)
    xmax = int(center[0] + w/2)
    ymin = int(center[1] - h/2)
    ymax = int(center[1] + h/2)
    
    if xmin < 0:
        xmin = 0 
    if ymin < 0:
        ymin = 0 

    new_box = [xmin,ymin,xmax,ymax]


    crop_frame = frame[ymin:ymax,xmin:xmax,:]
            
    crop_frame = cv2.resize(crop_frame,(192,256))
    
    return crop_frame, new_box


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    X_normalized =  X / w * 2 - [1, h / w]
    X_normalized = np.clip(X_normalized, -1, 1)
    return X_normalized



def get_sample(sample):
    image, label = sample

    transform1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((360,640))
    ])
    
    transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    image = np.array(image)
    
    box = label["box"]
    center = torch.Tensor([0.5 * (box[0] + box[2]) / image.shape[1], 0.5 * (box[1] + box[3]) / image.shape[0]])
    
    
    original_image = image.copy()
    image, _ = crop_around_person(image, box)
    image = transform2(image)
    
    original_image = transform1(original_image)

    
    
    gaze = label["direction"]
    gaze_direction = gaze[None,:,:]/np.linalg.norm(gaze)
    keypoints_3d_gt = np.expand_dims(label['pose3d'], axis=0)
    keypoints_3d_gt = keypoints_3d_gt[:,:,:3]

    hip_3d = (keypoints_3d_gt[:,11:12,:]+keypoints_3d_gt[:,12:13,:])/2
    mean_eyes_3d = (keypoints_3d_gt[:,1:2,:]+keypoints_3d_gt[:,2:3,:])/2 
    gaze_point = mean_eyes_3d + gaze_direction*0.3
    keypoints_3d_gt = np.concatenate((keypoints_3d_gt,hip_3d),axis=1)
    keypoints_3d_gt = np.concatenate((keypoints_3d_gt,gaze_point),axis=1)

    keypoints_3d_gt[:, :, :] -= keypoints_3d_gt[:, -2:-1, :]
    keypoints_3d_gt[:, -2, :] = 0
    keypoints_3d_gt = torch.from_numpy(keypoints_3d_gt).float()

    keypoints_2d = label['pose2d'][:,:2]
    hip2d = (keypoints_2d[11:12]+keypoints_2d[12:13])/2
    mean_eyes = (keypoints_2d[1:2]+keypoints_2d[2:3])/2 
    keypoints_2d = np.concatenate((keypoints_2d,hip2d),axis=0)
    keypoints_2d = np.concatenate((keypoints_2d,mean_eyes),axis=0)
    keypoints_2d = normalize_screen_coordinates(keypoints_2d,960,540)
    keypoints_2d = torch.from_numpy(keypoints_2d).float()
    
    keypoints_2d_crop = label['pose2d_cp'][:,:2]
    hip2d_crop = (keypoints_2d_crop[11:12]+keypoints_2d_crop[12:13])/2
    mean_eyes_crop = (keypoints_2d_crop[1:2]+keypoints_2d_crop[2:3])/2
    keypoints_2d_crop = np.concatenate((keypoints_2d_crop,hip2d_crop),axis=0)
    keypoints_2d_crop = np.concatenate((keypoints_2d_crop,mean_eyes_crop),axis=0)
    keypoints_2d_crop = torch.from_numpy(keypoints_2d_crop).float()

    objects = label["embedding"]

    return objects, image, keypoints_3d_gt, keypoints_2d, keypoints_2d_crop, box
