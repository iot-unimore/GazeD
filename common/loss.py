"""
Creative Commons Attribution-NonCommercial ShareAlike 4.0 International License  https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
import math
from matplotlib.pyplot import bone
import torch.nn.functional as F


def mpjpe_diffusion_all_min(predicted, target, mean_pos=False, mask_pos=False):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    if not mean_pos:
        t = predicted.shape[1]
        h = predicted.shape[2]
        target = target.unsqueeze(1).unsqueeze(1).repeat(1, t, h, 1, 1, 1)
        errors = torch.norm(predicted - target, dim=len(target.shape)-1)
        if mask_pos:
            mask = torch.all(target == torch.tensor([-10, -10, -10],dtype=target.dtype,device="cuda:0"), dim=-1)
            errors[mask] = 0
        errors = rearrange(errors, 'b t h f n  -> t h b f n', )
        min_errors = torch.min(errors, dim=1, keepdim=False).values
        min_errors = min_errors.reshape(t, -1)
        min_errors = torch.mean(min_errors, dim=-1, keepdim=False)
        return min_errors
    else:
        t = predicted.shape[1]
        h = predicted.shape[2]
        mean_pose = torch.mean(predicted, dim=2, keepdim=False)
        target = target.unsqueeze(1).repeat(1, t, 1, 1, 1)
        
        errors = torch.norm(mean_pose - target, dim=len(target.shape) - 1)
        if mask_pos:
            mask = torch.all(target == torch.tensor([-10, -10, -10],dtype=target.dtype,device="cuda:0"), dim=-1)
            errors[mask] = 0
        errors = rearrange(errors, 'b t f n  -> t b f n', )
        errors = errors.reshape(t, -1)
        errors = torch.mean(errors, dim=-1, keepdim=False)
        return errors

    


def mpjpe_diffusion_median(predicted, target, mask_pos = False):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    t = predicted.shape[1]
    h = predicted.shape[2]
    mean_pose = torch.median(predicted, dim=2, keepdim=False)[0]
    target = target.unsqueeze(1).repeat(1, t, 1, 1, 1)
    
    errors = torch.norm(mean_pose - target, dim=len(target.shape) - 1)
    if mask_pos:
        mask = torch.all(target == torch.tensor([-10, -10, -10],dtype=target.dtype,device="cuda:0"), dim=-1)
        errors[mask] = 0
    errors = rearrange(errors, 'b t f n  -> t b f n', )
    errors = errors.reshape(t, -1)
    errors = torch.mean(errors, dim=-1, keepdim=False)
    return errors



def mpjpe_diffusion(predicted, target, mean_pos=False, mask_pos=False):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    if not mean_pos:
        t = predicted.shape[1]
        h = predicted.shape[2]
        target = target.unsqueeze(1).unsqueeze(1).repeat(1, t, h, 1, 1, 1)
        errors = torch.norm(predicted - target, dim=len(target.shape)-1)
        if mask_pos:
            mask = torch.all(target == torch.tensor([-10, -10, -10],dtype=target.dtype,device="cuda:0"), dim=-1)
            errors[mask] = 0
        errors = rearrange(errors, 'b t h f n  -> t h b f n', )
        errors = errors.squeeze(-2)
        errors = torch.mean(errors, dim=-1, keepdim=False)
        errors = torch.min(errors, dim=1, keepdim=False).values
        errors = torch.mean(errors, dim=-1, keepdim=False)
        return errors







def mean_angular_error(predicted, target, leye, reye, mode="avg"):  

    predicted = torch.mean(predicted,dim=2, keepdim=False)
    predicted = predicted.squeeze(1)


    #MAE 3D computation

    mean_eyes_t = (target[:,:,leye,:] + target[:,:,reye,:])/2
    mean_eyes_p = (predicted[:,:,leye,:] + predicted[:,:,reye,:])/2 
    a = predicted[:,:,-1,:]-mean_eyes_p
    b = target[:,:,-1,:]-mean_eyes_t
    modulus_a = torch.linalg.norm(a,dim=-1, keepdim=True)
    modulus_b =torch.linalg.norm(b,dim=-1, keepdim=True)
    
    a = a / modulus_a
    b = b / modulus_b   
    cos_theta = F.cosine_similarity(a,b, dim=-1)
    cos_theta = torch.clamp(cos_theta, min=-1.0, max=1.0)
    rad = torch.acos(cos_theta)
    MAE3D = rad * (180 / math.pi)
    
    #MAE 2D computation

    c = a[...,:2] / torch.linalg.norm(a[...,:2],dim=-1, keepdim=True)
    d = b[...,:2] / torch.linalg.norm(b[...,:2],dim=-1, keepdim=True) 
    cos_theta = F.cosine_similarity(c,d, dim=-1)
    cos_theta = torch.clamp(cos_theta, min=-1.0, max=1.0)
    rad = torch.acos(cos_theta)
    MAE2D = rad * (180 / math.pi)
    

    
    return torch.mean(MAE3D, dim=0), torch.mean(MAE2D, dim=0)   



