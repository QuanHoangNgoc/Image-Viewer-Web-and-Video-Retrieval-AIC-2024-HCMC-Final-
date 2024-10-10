import os
import pandas as pd
import subprocess
import datetime 

from natsort import natsorted
from glob import glob
from tqdm.auto import tqdm
########################################################################

AIC_videos_path 	= 'D:/AIC_Data_2024/Video_data'
AIC_keyframes_path = 'D:/AIC_Data_2024/Extracted_Keyframes'

if os.path.exists(AIC_keyframes_path):
    pass
else: 
    os.mkdir(AIC_keyframes_path)
    
lst_videos = ["Videos_L10", "Videos_L11", "Videos_L12"]

count_video=0
count_keyframe=0

for i in range(len(lst_videos)):

    videos_path = os.path.join(AIC_videos_path, lst_videos[i], "video")
    video_name = lst_videos[i].split("_")[-1]
    videos_id_lst = natsorted(os.listdir(videos_path))
    keyframes_video_path = os.path.join(AIC_keyframes_path, f'Keyframes_{video_name}')
    
    if not os.path.exists(keyframes_video_path):
        os.mkdir(keyframes_video_path)
    print(lst_videos[i])
    
    for vid_id in tqdm(videos_id_lst):
        
        video_id_path = os.path.join(videos_path, vid_id)
        id = vid_id.split(".")[0]
        keyframes_video_id_path = os.path.join(keyframes_video_path, id)
        
        if not os.path.exists(keyframes_video_id_path):
            os.mkdir(keyframes_video_id_path)
            
        frame_dist = 100            # Get 1 keyframes every 100 frames 
        filter = "select='not(mod(n,100))',setpts=N/FRAME_RATE/TB"
        terminal_cmd = f'C:/ffmpeg/ffmpeg.exe -i {video_id_path} -vf {filter} -vsync vfr {keyframes_video_id_path}/{id}_%d.jpg'
        
        subprocess.run(terminal_cmd, shell = True)
        