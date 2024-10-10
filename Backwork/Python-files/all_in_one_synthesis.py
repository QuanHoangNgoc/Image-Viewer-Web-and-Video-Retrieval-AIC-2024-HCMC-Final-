import pickle
import numpy as np 
import torch 
import os 

from natsort import natsorted 
from tqdm.auto import tqdm 


root_working = 'D:/PycharmProjects/pythonProject/AIC_2024'
path_AiO = os.path.join(root_working,'all-in-one')

path_AiO_file_X_features = os.path.join (path_AiO, 'keyframes_features_blip.npy')
path_AiO_file_lst_features = os.path.join (path_AiO, 'keyframes_features_blip.pkl')

original_feature_path = os.path.join(root_working, 'keyframes_features_blip')

pkl_content = [] 
npy_content = [] 

video_lst = natsorted(os.listdir(original_feature_path))

for video in tqdm(video_lst):
    video_path = os.path.join(original_feature_path, video)
    video_id_lst = natsorted(os.listdir(video_path))
    for video_id in (video_id_lst):
        try:
            with open(os.path.join(video_path, video_id), "rb") as file:
                video_id_ft = pickle.load(file)
                
            values = list(video_id_ft.values())
            values_np = [tensor.numpy() for tensor in video_id_ft.values()]
            keys = list(video_id_ft.keys())
            numpy_arrays = np.array([value.numpy() for value in values])
            
            npy_content.extend(values_np)
            pkl_content.extend(keys)
        except:
            print(video_id)

print(len(npy_content))
print(len(pkl_content))
npy_content = np.array(npy_content)
print(len(npy_content))
np.save(path_AiO_file_X_features, npy_content)

with open(path_AiO_file_lst_features, "wb") as f: 
    pickle.dump(pkl_content, f)

