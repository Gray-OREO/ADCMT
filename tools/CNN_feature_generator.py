import torch
from torchvision import transforms, models
from torchvision.models.resnet import ResNet50_Weights
import torch.nn as nn
from torch.utils.data import Dataset
import skvideo.io
from PIL import Image
import os
import h5py
import numpy as np
import random
import pandas as pd
from argparse import ArgumentParser
import chardet


class VideoDataset(Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, videos_dir, video_names, score):

        super(VideoDataset, self).__init__()
        self.videos_dir = videos_dir
        self.video_names = video_names
        self.score = score

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name))
        video_score = self.score[idx]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        video_length = video_data.shape[0]
        video_channel = video_data.shape[3]
        video_height = video_data.shape[1]
        video_width = video_data.shape[2]
        transformed_video = torch.zeros([video_length, video_channel, video_height, video_width])
        for frame_idx in range(video_length):
            frame = video_data[frame_idx]
            frame = Image.fromarray(frame)
            frame = transform(frame)
            transformed_video[frame_idx] = frame

        sample = {'video': transformed_video,
                  'score': video_score,
                 'name':video_name}

        return sample


class ResNet50(torch.nn.Module):
    """Modified ResNet50 for feature extraction"""

    def __init__(self):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(*list(models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).children())[:-2])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        # features@: 7->res5c(4096) 6->res4c(2048) 5->res3c(1024) 4->res2c(512)
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == 7:
                features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
                features_std = global_std_pool2d(x)
                return features_mean, features_std


def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)


def get_features(video_data, frame_batch_size=32, device='cuda'):
    """feature extraction"""
    extractor = ResNet50().to(device)
    video_length = video_data.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    extractor.eval()
    with torch.no_grad():
        while frame_end < video_length:
            batch = video_data[frame_start:frame_end].to(device)
            features_mean, features_std = extractor(batch)
            output1 = torch.cat((output1, features_mean), 0)
            output2 = torch.cat((output2, features_std), 0)
            frame_end += frame_batch_size
            frame_start += frame_batch_size

        last_batch = video_data[frame_start:video_length].to(device)
        features_mean, features_std = extractor(last_batch)
        output1 = torch.cat((output1, features_mean), 0)
        output2 = torch.cat((output2, features_std), 0)
        output = torch.cat((output1, output2), 1).squeeze()

    return output


if __name__ == "__main__":
    parser = ArgumentParser(description='"Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50')
    parser.add_argument("--seed", type=int, default=19980427)
    parser.add_argument('--database', default='LSVQs', type=str,
                        help='database name (default: KoNViD-1k)')
    parser.add_argument('--frame_batch_size', type=int, default=32,
                        help='frame batch size for feature extraction (default: 16)')

    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    args = parser.parse_args([])

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    if args.database == 'KoNViD-1k':
        videos_dir = 'E:/Gray/Database/KoNViD_1k/KoNViD_1k_videos/'  # videos dir
        features_dir = 'E:/Gray/Database/KoNViD_1k/CNN_features_KoNViD-1k_res5c/'  # features dir
        datainfo = '../data/KoNViD-1kinfo.mat'  # database info: video_names, scores; video format, width, height, index, ref_ids, max_len, etc.
        data = pd.read_csv("E:/Gray/Database/KoNViD_1k/KoNViD_1k_metadata/KoNViD_1k_attributes.csv")
        video_id = data["flickr_id"].tolist()
        video_names = [str(i)+".mp4" for i in video_id]
    if args.database == 'CVD2014':
        videos_dir = '/data/Gray/Database/CVD2014/'
        features_dir = 'CNN_features_CVD2014_res3c/'
        datainfo = 'data/CVD2014info.mat'
    if args.database == 'LIVE-VQC':
        videos_dir = 'G:/Databse/LIVE-VQC/Video/'
        features_dir = 'G:/Database/LIVE-VQC/CNN_features_LIVE-VQC/'
        datainfo = '../data/LIVE-VQCinfo.mat'
    if args.database == 'LSVQs':
        videos_dir = 'G:/Database/LSVQs/videos/'
        features_dir = 'G:/Database/LSVQs/CNN_features_LSVQs/'
        datainfo = '../data/LSVQsinfo.mat'

    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")
    Info = h5py.File(datainfo, 'r')

    if args.database != 'KoNViD-1k':
        video_names = [Info[Info['video_names'][0, :][i]][()].tobytes()[::2].decode() for i in range(len(Info['video_names'][0, :]))]

    scores = Info['scores'][0, :]
#     video_format = Info['video_format'][()].tobytes()[::2].decode()
#     width = int(Info['width'][0])
#     height = int(Info['height'][0])
    dataset = VideoDataset(videos_dir, video_names, scores)

    for i in range(len(dataset)):
        current_data = dataset[i]
        current_name = current_data['name']
        current_video = current_data['video']
        current_score = current_data['score']
        print('Video {}:{} length {} Processing'.format(i, current_name, current_video.shape[0]))
        # print(current_video.shape)
        features = get_features(current_video, args.frame_batch_size, device)
        np.save(features_dir + str(i) + '_resnet-50_res5c', features.to('cpu').numpy())
        np.save(features_dir + str(i) + '_score', current_score)
