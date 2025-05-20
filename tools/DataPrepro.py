import skvideo.io
import numpy as np
import pandas as pd
import h5py
import os

# #KoNViD_1k & LIVE_VQC 实例测试
# video_dir = ['E:/Gray/Database/KoNViD_1k/KoNViD_1k_videos',
#              'E:/Gray/Database/VQA/LIVE Video Quality Challenge (VQC) Database/Video']
# video_name = ['2999049224.mp4', 'A001.mp4']
# video_data = skvideo.io.vread(os.path.join(video_dir[1], video_name[1]))  # [T,H,W,C]
# np.save(r'A001.npy', video_data)  # 1.59M -> 355M   23.9M -> 1.73G

videos_dir = 'E:/Gray/Database/KoNViD_1k/KoNViD_1k_videos/'  # videos dir
datainfo = 'data/KoNViD-1kinfo.mat'
# database info: video_names, scores; video format, width, height, index, ref_ids, max_len, etc.
if not os.path.exists('KoNViD_1k_npy'):
    os.makedirs('KoNViD_1k_npy')
Info = h5py.File(datainfo, 'r')
data = pd.read_csv('E:/Gray/DataBase/KoNViD_1k/KoNViD_1k_metadata/KoNViD_1k_attributes.csv')
video_ids = data['flickr_id'].tolist()
# video_names = [str(i) + ".mp4" for i in video_id]
i = 1
for video_id in video_ids:
    video_data = skvideo.io.vread(os.path.join(videos_dir, str(video_id) + ".mp4"))
    np.save(r'KoNViD_1k_npy/{}.npy'.format(video_id), video_data)
    print('\rData Preprocessing:{}/{}'.format(i, len(video_ids)), end='')
    i += 1
print('Data Preprocessing Over!')