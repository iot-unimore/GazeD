"""
Creative Commons Attribution-NonCommercial ShareAlike 4.0 International License  https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

from transformers import RTDetrForObjectDetection, RTDetrConfig
import os
import joblib
import cv2
from torchvision import transforms
import numpy as np
import torch
from tqdm import tqdm

"""
This script is used to compute object embeddings on the dataset. These sample is for GFIE. 
Adjust the paths accordingly for the other datasets.
"""

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((640,640))
])

configuration = RTDetrConfig(num_queries=100)
rtdetr = RTDetrForObjectDetection(config=configuration).from_pretrained("PekingU/rtdetr_r50vd").to("cuda:0")

if not os.path.exists(f"GFIE/embeddings"):
    os.mkdir(f"GFIE/embeddings")


for t in ["train", "valid", "test"]:
    
    if not os.path.exists(f"GFIE/embeddings/{t}"):
        os.mkdir(f"GFIE/embeddings/{t}")
    
    for s in tqdm(os.listdir(f"GFIE/rgb/{t}")):
        
        if not os.path.exists(f"GFIE/embeddings/{t}/{s}"):
            os.mkdir(f"GFIE/embeddings/{t}/{s}")
        

        folder = [i for i in os.listdir(f"rgb/train/{s}") if i.endswith(".jpg")] 
        for i in tqdm(folder):

            image = cv2.imread(f"GFIE/rgb/{t}/{s}/"+i)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = transform(image).unsqueeze(0).to("cuda:0")
            with torch.no_grad():
                results = rtdetr(image)
            embed = results['last_hidden_state'].squeeze(0).detach().cpu().numpy()
            np.save(f"GFIE/embeddings/{t}/{s}/{i.removesuffix('.jpg')}.npy", embed)
