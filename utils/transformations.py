"""
Creative Commons Attribution-NonCommercial ShareAlike 4.0 International License  https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import numpy as np
import cv2
import torch

def get_bb(pose2d, image_size):
     
    x_min = max(0,float(np.min(pose2d[:, 0]))) 
    y_min = max(0,float(np.min(pose2d[:, 1]))) 
    x_max = min(image_size[1],float(np.max(pose2d[:, 0])))
    y_max = min(image_size[0],float(np.max(pose2d[:, 1])))


    boxes = []
    boxes.append([x_min, y_min, x_max , y_max])
    
    return np.array(boxes)

def expand_box_and_crop(frame, box):
    center = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))
    scale = ((box[2] - box[0]), (box[3] - box[1]))

    h = scale[1]*1.2
    w = 192/256*h

    xmin = int(center[0] - w/2)
    xmax = int(center[0] + w/2)
    ymin = int(center[1] - h/2)
    ymax = int(center[1] + h/2)
    
    if xmin < 0:
        xmin = 0 

    new_box = [xmin,ymin,xmax,ymax]
    


    crop_frame = frame[ymin:ymax,xmin:xmax,:]
            
    crop_frame = cv2.resize(crop_frame,(192,256))
    
    return crop_frame, new_box

def get_affine_transform(center, scale, rotation, output_size):
   
    w, h = output_size
    src_w = scale[0]
    src_h = scale[1]

    ymax = center[1] + src_h / 2

    src_points = np.array([
        [center[0] - src_w / 2, center[1] - src_h / 2], 
        [center[0] + src_w / 2, center[1] - src_h / 2],  
        [center[0] - src_w / 2, ymax]
    ], dtype=np.float32)


    dst_points = np.array([
        [0, 0],       
        [w, 0],      
        [0, h]
    ], dtype=np.float32)


    if rotation != 0:   
        rotation_matrix = cv2.getRotationMatrix2D((center[0], center[1]), rotation, 1)
        src_points = cv2.transform(src_points[None, :, :], rotation_matrix)[0]
    trans = cv2.getAffineTransform(src_points, dst_points)

    return trans

def crop_image(image, center, scale, output_size):
	trans = get_affine_transform(center, scale, 0, output_size)
	image = cv2.warpAffine(image, trans, (output_size), flags=cv2.INTER_LINEAR)

	return image

def crop_image2(image, bounding_box, size=(192, 256)):
    x_min, y_min, x_max, y_max = bounding_box

 
    height, width = image.shape[:2]
    x_min = max(0, int(x_min))
    y_min = max(0, int(y_min))
    x_max = min(width, int(x_max))
    y_max = min(height, int(y_max))

   
    cropped_image = image[y_min:y_max, x_min:x_max]

    resized_image = cv2.resize(cropped_image, size)

    return resized_image

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def crop_pose2d(pose2d, box):

    keypoints2D_crop = np.zeros([pose2d.shape[0], 2], dtype='float32')

    center = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))
    scale = ((box[2] - box[0]), (box[3] - box[1]))

    trans = get_affine_transform(center, scale, 0, [192, 256])

    for j in range(pose2d.shape[0]):
        keypoints2D_crop[j] = affine_transform(pose2d[j], trans)
        
    return keypoints2D_crop

def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    X_normalized =  X / w * 2 - [1, h / w]
    X_normalized = np.clip(X_normalized, -1, 1)
    return X_normalized

def extract_gaze_direction(predicted_3d_pose,reye,leye):
    
    
    pose3d = predicted_3d_pose[:,0,:,:]
   
    gaze = pose3d[:,-1:,:]
    eye = (pose3d[:,reye:reye+1,:]+pose3d[:,leye:leye+1,:])/2
    
    a = gaze-eye
    modulus_a = np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8
    
    directions = a/modulus_a
    
    
    return directions