"""
Creative Commons Attribution-NonCommercial ShareAlike 4.0 International License  https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import numpy as np
import torch.nn.functional as F
import torch


def mean_angular_error(predicted, target, leye=15, reye=17, mode="avg", oracle_mode='gaze'):  

    if isinstance(predicted, np.ndarray):
        predicted = torch.from_numpy(predicted)
    
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)

    # b,h,j,c
    if mode=="avg":
        predicted = torch.mean(predicted, dim=1, keepdim=True)
    elif mode=="median":
        predicted,_ = torch.median(predicted,dim=1, keepdim=True)
    elif mode=="oracle_gaze":
        errors = torch.norm(predicted[:,:,-1:,:] - target[:,:,-1:,:], dim=len(target.shape)-1)
        distance = torch.mean(errors, dim=-1, keepdim=False)
        indices = torch.argmin(distance, dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        indices = indices.expand(-1, 1, 20, 3)
        predicted = torch.gather(predicted,1,indices)
    elif mode=="oracle_skel":
        errors = torch.norm(predicted[:,:,:,:] - target[:,:,:,:], dim=len(target.shape)-1)
        distance = torch.mean(errors, dim=-1, keepdim=False)
        indices = torch.argmin(distance, dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        indices = indices.expand(-1, 1, 20, 3)
        predicted = torch.gather(predicted,1,indices)
    elif mode=="oracle_joint":
        errors = torch.norm(predicted[:,:,:,:] - target[:,:,:,:], dim=len(target.shape)-1) # error on the joint
        indices = torch.argmin(errors, dim=1).unsqueeze(1).unsqueeze(3)
        indices = indices.expand(-1, 1, -1, 3)
        predicted = torch.gather(predicted,1,indices)
    
    #MAE 3D computation

    mean_eyes_t = (target[:,:,leye,:] + target[:,:,reye,:])/2
    mean_eyes_p = (predicted[:,:,leye,:] + predicted[:,:,reye,:])/2 
    a = predicted[:,:,-1,:]-mean_eyes_p
    b = target[:,:,-1,:]-mean_eyes_t  
    
    a = a/torch.linalg.norm(a, dim=-1, keepdim=True)
    b = b/torch.linalg.norm(b, dim=-1, keepdim=True)
    cos_theta = F.cosine_similarity(a,b, dim=-1) 
    
    cos_theta = torch.clamp(cos_theta, min=-1.0, max=1.0)
    rad = torch.acos(cos_theta)
    MAE3D = rad * (180 / torch.pi)
    
    #MAE 2D computation
    c = a[...,:2] / torch.linalg.norm(a[...,:2],dim=-1, keepdim=True)
    d = b[...,:2] / torch.linalg.norm(b[...,:2],dim=-1, keepdim=True)

    
    cos_theta = F.cosine_similarity(c,d, dim=-1) 
    cos_theta = torch.clamp(cos_theta, min=-1.0, max=1.0)
    rad = torch.acos(cos_theta)
    MAE2D = rad * (180 / torch.pi)
    
    return torch.mean(MAE3D).item(), torch.mean(MAE2D).item()


"""
gt = np.load("gfie_inf/gt.npy")
pred = np.load("gfie_inf/predictions.npy")

MAE3D, MAE2D = mean_angular_error(pred, gt, 15, 17, mode="avg")

print(MAE3D, MAE2D)
"""