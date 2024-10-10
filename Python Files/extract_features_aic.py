import os
import subprocess
import numpy as np
import sys 
import pandas as pd
import pickle
import torch
from natsort import natsorted 
from torch import nn
from glob import glob

from tqdm.auto import tqdm
from PIL import Image

device = f"cuda:0" if torch.cuda.is_available() else "cpu"

root_working = "/home/dungmt"

keyframes_folder_lst = ["Keyframes_L01", "Keyframes_L02", "Keyframes_L03", "Keyframes_L04", "Keyframes_L05", "Keyframes_L06", "Keyframes_L07", "Keyframes_L08", "Keyframes_L09", "Keyframes_L10", "Keyframes_L11", "Keyframes_L12"]
keyframes_features_path = os.path.join(root_working, "keyframes_features_beit3")
if not os.path.exists(keyframes_features_path):
    os.mkdir(keyframes_features_path)

sys.path.insert(0,"/mlcv1/WorkingSpace/Personal/thuyentd/trecvid/src/beit3/lib/BEiT3")
from BEiT3.beit3 import load_model, encode_image, encode_text  
print('Loading BEiT-3 model...')
model, processor, tokenizer = load_model(
    device=device, 
    checkpoint="/mlcv1/WorkingSpace/Personal/thuyentd/trecvid/src/beit3/checkpoints/beit3_large_patch16_384_coco_retrieval.pth", 
    sentencepiece_model="/home/dungmt/beit/checkpoints/beit3.spm",
    input_size=384,
    model_name="beit3_large_patch16_384",
)
print("Load BEiT-3 model: OK")

'''
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode 
sys.path.insert(0,"/home/dungmt/BLIP/")
from models.blip_itm import blip_itm
os.chdir("./BLIP/") 
image_size = 384
transform = transforms.Compose([
    transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]) 
print("Loading BLIP model...")
model = blip_itm(pretrained="/home/dungmt/BLIP/checkpoints/blip_model_large_retrieval_coco.pth", image_size=image_size, vit='large')
model.eval()
model.to(device)
print("Load BLIP model: Ok")
'''

for kf_folder in keyframes_folder_lst:

    keyframes_folder_path = os.path.join(root_working, kf_folder)
    keyframes_feature_path = os.path.join(keyframes_features_path, kf_folder)
    video_id_lst = os.listdir(keyframes_folder_path)
    
    if not os.path.exists(keyframes_feature_path):
        os.mkdir(keyframes_feature_path)
    print(kf_folder)
    for video_id in tqdm(video_id_lst):
        kf_video_id_path = os.path.join(keyframes_folder_path, video_id)
        kf_video_id_ft_path = os.path.join(keyframes_feature_path, video_id)
        kf_video_id_ft_file = f"{kf_video_id_ft_path}.pkl"
        
        if not os.path.exists(kf_video_id_ft_file):
            with open(kf_video_id_ft_file, "wb") as f:
                pass
            
        kf_lst = natsorted(os.listdir(kf_video_id_path))
        features = {}
        
        for kf in kf_lst:
            kf_path = os.path.join(kf_video_id_path, kf)
            
            # blip 
            '''
            image = transform(Image.open(kf_path)).unsqueeze(0).to(device)
            ft = model.get_feature(image, mode="image")
            '''
            
            # beit-3
            image = Image.open(kf_path) 
            ft = encode_image(
                model=model,
                processor=processor, 
                image=[image],
                device=device,
                return_torch=True,
            )            

            ft = ft.detach().cpu()
            features[kf] = ft.squeeze() 
            
        with open(kf_video_id_ft_file, "wb") as f:
            pickle.dump(features, f)
'''
# Extract text feature using BEiT-3
text_query_ft = encode_text(model=model,  
                            tokenizer=tokenizer, 
                            text=query, 
                            device=device, 
                            return_torch=True,
                            )
'''