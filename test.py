"""
Creative Commons Attribution-NonCommercial ShareAlike 4.0 International License  https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import errno
import os
import random
import sys
import warnings
from time import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import webdataset as wds
from accelerate import Accelerator
from common.arguments import parse_args
from common.diff_pipeline import DiffPipeGaze
from common.loss import (mpjpe_diffusion, mpjpe_diffusion_all_min,
                         mean_angular_error)
from common.gaze_dformer import GazeTransformer
from diffusers import DDIMScheduler, DDPMScheduler
from mvn.models import pose_hrnet
from mvn.utils.cfg import config, update_config
from tqdm import tqdm


warnings.simplefilter(action='ignore', category=FutureWarning)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


args = parse_args()

print('python ' + ' '.join(sys.argv))
print("CUDA Device Count: ", torch.cuda.device_count())
print(args)
print(f"GPU : {torch.cuda.get_device_name(0)}")

# set random seed
manualSeed = 1234
random.seed(manualSeed)
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(manualSeed)

# create checkpoint directory if it does not exist
if args.checkpoint=='':
    raise ValueError('Invalid checkpoint path')
try:
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

update_config(args.config)

accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="no")

print("Loading Dataset...")


if args.dataset.lower() == "gfie":
    from mvn.datasets.gfie import get_sample
elif args.dataset.lower() == "gafa":
    from mvn.datasets.gafa import get_sample
elif args.dataset.lower() == "egoexo":
    from mvn.datasets.egoexo import get_sample


test_dataset = wds.WebDataset(config.dataset.root+'/test/'+config.dataset.test_shards, shardshuffle=False)
test_dataset = test_dataset.decode('pil').to_tuple('jpg', 'pyd').map(get_sample)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, 
                                                num_workers=1, worker_init_fn=seed_worker)


dataset_name = args.dataset.lower()
num_joints = config.gaze.num_joints
root_joint = config.gaze.root_joint
leye = config.gaze.leye
reye = config.gaze.reye

# define backbone
backbone = pose_hrnet.get_pose_net(config.model.backbone)
ret = backbone.load_state_dict(torch.load(f"{args.checkpoint}/posehrnet/pose_hrnet_w32_256x192.pth"), strict=False)
print("Loading backbone from {}".format(f"{args.checkpoint}/posehrnet/pose_hrnet_w32_256x192.pth"))
print("Backbone weights are fixed!")
for p in backbone.parameters():
    p.requires_grad = False
backbone.eval()


model_pos_val = GazeTransformer(num_joints=num_joints, 
                                drop_path_rate=0)

        
# set noise schedulers
noise_scheduler = DDPMScheduler(num_train_timesteps=100)
noise_scheduler.config.prediction_type = "epsilon"
val_scheduler = DDIMScheduler(num_train_timesteps=100)
val_scheduler.set_timesteps(args.timesteps, device=accelerator.device)
val_scheduler.config.prediction_type = "epsilon"

# prepare model and dataloader with accelerator
backbone = accelerator.prepare_model(backbone)
model_pos_val = accelerator.prepare_model(model_pos_val)


# put models on gpu if available
if torch.cuda.is_available():
    backbone = backbone.to(accelerator.device)
    model_pos_val = model_pos_val.to(accelerator.device)

# resume training from pretrained model
if args.evaluate:
    chk_filename = os.path.join(args.checkpoint+f"/model_{dataset_name.lower()}_test/", args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    checkpoint_model = checkpoint['model_pos']
    model_pos_val.load_state_dict(checkpoint_model, strict=False)
    




model_pos_val.eval()
model_pos_val = accelerator.unwrap_model(model_pos_val)

pipeline = DiffPipeGaze(model_pos_val, val_scheduler).to(accelerator.device)

with torch.no_grad():
    mpjpe_avg_hypothesis = 0
    mpjpe_best_skel_hypothesis = 0
    mean_angular_error_hypothesis = 0
    mean_gaze_error_hypothesis = 0
    mean_mae2d = 0
    pred_skeletons = []
    gt_skeletons = []
    MAE3D = 0
    total_elements = 0
    it = 0
    
    
    # evaluate on test set
    for object_embeddings, images, inputs_3d, inputs_2d, inputs_2d_crop , center in test_dataloader:
        start = time()

        
        b, _, _, _ = images.shape
        
        img = torch.clone(images)

        images = images.to(accelerator.device)
        images = images.view(b, images.shape[1], images.shape[2], images.shape[3])
        inputs_2d = inputs_2d.to(accelerator.device)
        poses2d = torch.clone(inputs_2d_crop)
        inputs_2d = inputs_2d.view(b, inputs_2d.shape[1], inputs_2d.shape[2])[:, None]
        inputs_2d_crop = inputs_2d_crop.to(accelerator.device)
        inputs_2d_crop = inputs_2d_crop.view(b, inputs_2d_crop.shape[1], inputs_2d_crop.shape[2])
        center = center.to(accelerator.device)
        inputs_3d = inputs_3d.to(accelerator.device)
        object_embeddings = object_embeddings.to(accelerator.device)


        # reference points for deformable attention on backbone features
        inputs_2d_crop[..., :2] /= torch.tensor([192//2, 256//2], device=images.device)
        inputs_2d_crop[..., :2] -= torch.tensor([1, 1], device=images.device)

        
        shape = (b, args.num_proposals, 1, num_joints, 3)
        random_noise = torch.randn(shape).to(accelerator.device)
        
        
        context_features = backbone(images)

        predicted_3d_pos = pipeline(random_noise, inputs_2d, inputs_2d_crop, context_features, object_embeddings, center)

        predicted_3d_pos[:, :, :, :, root_joint, :] = 0
    
        
        batch_mpjpe_avg_hypothesis = mpjpe_diffusion_all_min(predicted_3d_pos[:, -1:,:,:,:-1,:], inputs_3d[:,:,:-1,:], mean_pos=True)
        batch_mpjpe_best_skel_hypothesis = mpjpe_diffusion(predicted_3d_pos[:, -1:,:,:,:-1,:], inputs_3d[:,:,:-1,:])
        batch_error_gaze = mpjpe_diffusion_all_min(predicted_3d_pos[:, -1:,:,:,-1:,:], inputs_3d[:,:,-1:,:], mean_pos=True)
        batch_min_gaze = mpjpe_diffusion(predicted_3d_pos[:, -1:,:,:,-1:,:], inputs_3d[:,:,-1:,:], mean_pos=False)
        batch_mean_angular_error_hypothesis,batch_mae2d= mean_angular_error(predicted_3d_pos[:, -1:], inputs_3d, reye, leye)

        results = {}                    
            
        mpjpe_avg_hypothesis += batch_mpjpe_avg_hypothesis.item()*b
        mpjpe_best_skel_hypothesis += batch_mpjpe_best_skel_hypothesis.item()*b
        mean_gaze_error_hypothesis += batch_error_gaze.item()*b
        mean_angular_error_hypothesis += batch_mean_angular_error_hypothesis.item()*b
        mean_mae2d += batch_mae2d.item()*b
        
        MAE3D += batch_mean_angular_error_hypothesis.item()*b
        total_elements += b

        pred_skeletons.append(predicted_3d_pos[:, -1, :, 0, :, :].detach().cpu())
        gt_skeletons.append(inputs_3d.detach().cpu())
        
        print(f"[{it}] ({time() - start:.2f} seconds)"
                f"MAE3D: {mean_angular_error_hypothesis / total_elements:.2f} ° |"
                f"MAE2D: {mean_mae2d/ total_elements:.2f} ° |"
                f"MPJPE (Avg): {mpjpe_avg_hypothesis * 1000 / total_elements:.2f} mm |"
                f"MPJPE (Best Skeleton): {mpjpe_best_skel_hypothesis * 1000 / total_elements:.2f} mm |"
                f"MPJPE (Avg Gaze): {mean_gaze_error_hypothesis * 1000 / total_elements:.2f} mm |")

        it += 1
    
    if args.save_predictions:  #If you want to save the predictions to make some offline evaluations
        
        save_dir = f'predictions/{dataset_name}_inf'

        if os.path.exists(save_dir) == False:
            os.mkdir(save_dir)

        np.save(f'{save_dir}/predictions.npy', np.vstack(pred_skeletons))
        np.save(f'{save_dir}/gt.npy', np.vstack(gt_skeletons))

    mpjpe_avg_hypothesis = mpjpe_avg_hypothesis / total_elements
    mpjpe_best_skel_hypothesis = mpjpe_best_skel_hypothesis / total_elements
    mean_gaze_error_hypothesis = mean_gaze_error_hypothesis / total_elements
    mean_angular_error_hypothesis = mean_angular_error_hypothesis / total_elements
    mean_mae2d = mean_mae2d / total_elements
    

    print(f"Average MAE3D: {mean_angular_error_hypothesis:.2f} ° ")
    print(f"Average MAE2D: {mean_mae2d:.2f} ° ")
    print(f"Average MPJPE (Gaze): {mean_gaze_error_hypothesis * 1000:.2f} mm \n")
    print(f'Average MPJPE (Avg Hypothesis): {mpjpe_avg_hypothesis * 1000:.2f} mm \n')
    print(f'MPJPE (Best Per-Skeleton Hypothesis): {mpjpe_best_skel_hypothesis * 1000:.2f} mm \n')
 
    
