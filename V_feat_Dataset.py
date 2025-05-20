from torch.utils.data import Dataset
import numpy as np
from tools.Score2Rank import MS_indices_for_levels_init


class VQARDataset(Dataset):
    def __init__(self, features_dir='CNN_features_KoNViD-1k/', index=None, max_len=240, feat_dim=4096, scale=1):
        super(VQARDataset, self).__init__()
        self.features = np.zeros((len(index), max_len, feat_dim))
        self.length = np.zeros((len(index), 1))
        self.mos = np.zeros((len(index), 1))
        for i in range(len(index)):
            features = np.load(features_dir + str(index[i]) + '_resnet-50_res5c.npy')
            self.length[i] = features.shape[0]
            if features.shape[0] > max_len:
                self.features[i, :features.shape[0], :] = features[:max_len]
            else:
                self.features[i, :features.shape[0], :] = features
            self.mos[i] = np.load(features_dir + str(index[i]) + '_score.npy')  #
        self.scale = scale  #
        self.label = self.mos / self.scale  # label normalization
        self.cls = MS_indices_for_levels_init(self.mos)

    def __len__(self):
        return len(self.mos)

    def __getitem__(self, idx):
        sample = self.features[idx], self.length[idx], self.label[idx], self.cls[idx]
        return sample