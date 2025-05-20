import h5py
# import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans, estimate_bandwidth, MeanShift
import numpy as np


def sorted_level(x, sorted_kc_l):
    # 根据质量分数等级定义level标签
    if x == sorted_kc_l[0]:
        return 0
    elif x == sorted_kc_l[1]:
        return 1
    else:
        return 2


def KM_indices_for_levels(MOSs, seed=123, rank=3):
    MOSs = MOSs.reshape(-1,1)
    km = KMeans(n_clusters=rank, random_state=seed)
    c = km.fit(MOSs)
    RankIndices = [[] for _ in range(rank)]
    for i, x in enumerate(c.labels_):
        RankIndices[x].append(i)
    return RankIndices


def MS_indices_for_levels(MOSs, quantile=0.3):
    bw = estimate_bandwidth(MOSs, n_samples=len(MOSs), quantile=quantile)
    model = MeanShift(bandwidth=bw, bin_seeding=True)
    c = model.fit(MOSs)
    classes = model.cluster_centers_.shape[0]
    RankIndices = [[] for _ in range(classes)]
    for i, x in enumerate(c.labels_):
        RankIndices[x].append(i)
    return RankIndices


def MS_indices_for_levels_init(MOSs, quantile=0.3):
    bw = estimate_bandwidth(MOSs, n_samples=len(MOSs), quantile=quantile)
    model = MeanShift(bandwidth=bw, bin_seeding=True)
    c = model.fit(MOSs)
    cls = []
    for i in c.labels_:
        cls.append(i)
    return cls


if __name__ == '__main__':
    datainfo = ['../data/KoNViD-1kinfo.mat', '../data/LIVE-VQCinfo.mat']  # database info: video_names, scores; video format, width, height, index, ref_ids, max_len, etc.
    Info = h5py.File(datainfo[0], 'r')
    scores = Info['scores'][0, :].reshape(-1, 1)
    print(scores)
    RankList = MS_indices_for_levels_init(scores, quantile=0.1)
    print(len(RankList))

