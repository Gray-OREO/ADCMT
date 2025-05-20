from argparse import ArgumentParser
import os
import h5py
import torch
from torch.optim import Adam, lr_scheduler, SGD
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset
# from Dataset import VQADataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import random
from scipy.stats import kendalltau
from tensorboardX import SummaryWriter
from models.ADCMT import ADCMT
import datetime
import time
from V_feat_Dataset import VQARDataset
from scipy import stats
from tools.CDCR_loss import mini_CDCR_loss, mini_ContraLoss
from tools.get_more import get_features

# from ranger import Ranger  # this is from ranger.py
# from ranger import RangerVA  # this is from ranger913A.py
# from ranger import RangerQH  # this is from rangerqh.py

parser = ArgumentParser(description='ADCMT')
parser.add_argument("--seed", type=int, default=19980427)  # 42
parser.add_argument('--database', default='LSVQs', type=str,
                    help='database name (default: KoNViD-1k or CVD2014 or LIVE-Qualcomm)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='frame batch size for feature extraction (default: 32)')
parser.add_argument('--disable_gpu', action='store_true',
                    help='flag whether to disable GPU')
parser.add_argument('--exp_id', default=6, type=int,
                    help='exp id for train-val-test splits (default: 0)')  # K:2
parser.add_argument('--test_ratio', type=float, default=0.2,
                    help='test ratio (default: 0.2)')
parser.add_argument('--val_ratio', type=float, default=0.2,
                    help='val ratio (default: 0.2)')
parser.add_argument('--deviceID', type=int, default=0,
                    help='device for GPU (default: 1)')
parser.add_argument('--model', default='ADCMT', type=str,
                    help='model name (default: ADCMT)')
parser.add_argument('--lr', type=float, default=0.00001,
                    help='learning rate (default: 0.00001)')
parser.add_argument("--notest_during_training", action='store_true',
                    help='flag whether to test during training')
parser.add_argument("--disable_visualization", action='store_true',
                    help='flag whether to enable TensorBoard visualization')
parser.add_argument('--weight_decay', type=float, default=0.,
                    help='weight decay (default: 0.0)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 200)')
parser.add_argument("--log_dir", type=str, default="logs",
                    help="log directory for Tensorboard log output")
parser.add_argument("--accumu_steps", type=int, default=1,
                    help='Real_batch_size = batch_size * accumu_steps')
parser.add_argument("--frame_num", type=int, default=240,
                    help='Training sample generation number')
parser.add_argument("--beta", type=float, default=0.5,
                    help='Coefficent of CDCR_loss')
parser.add_argument("--width", type=int, default=3,
                    help='width of ADCMT')
parser.add_argument("--depth", type=int, default=7,
                    help='depth of ADCMT')
parser.add_argument("--tau", type=float, default=0.7,
                    help='temperature')
parser.add_argument("--dropout", type=float, default=0.,
                    help='temperature')
parser.add_argument("--emb_dropout", type=float, default=0.,
                    help='temperature')
args = parser.parse_args()  # args=[]非命令行传参时加入

args.decay_interval = int(args.epochs / 10)
args.decay_ratio = 0.2

torch.manual_seed(args.seed)  #
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
random.seed(args.seed)

torch.utils.backcompat.broadcast_warning.enabled = True

if args.database == 'KoNViD-1k':
    features_dir = 'E:/Gray/Database/KoNViD_1k/CNN_features_KoNViD-1k_res5c/'  # features dir
    datainfo = 'data/KoNViD-1kinfo.mat'

elif args.database == 'CVD2014':
    features_dir = '/data/Gray/Database/CVD2014/CNN_features_CVD2014_res5c/'
    datainfo = 'data/CVD2014info.mat'

elif args.database == 'LIVE-VQC':
    features_dir = 'G:/Database/LIVE-VQC/CNN_features_LIVE-VQC/'
    datainfo = 'data/LIVE-VQCinfo.mat'

elif args.database == 'LIVE-Qualcomm':
    features_dir = '/data/Gray/Database/LIVE-Qualcomm/CNN_features_LIVE-Qualcomm_res5c/'
    datainfo = 'data/LIVE-Qualcomminfo.mat'

elif args.database == 'YT-UGC':
    features_dir = 'E:/Gray/Database/YT-UGC/CNN_features_YTUGC/'
    datainfo = 'data/YT-UGCinfo.mat'

elif args.database == 'LSVQs':
    features_dir = 'G:/Database/LSVQs/CNN_features_LSVQs/'
    datainfo = 'data/LSVQsinfo.mat'

else:
    raise ValueError('Unsupported database: {}'.format(args.database))

print('EXP ID: {}'.format(args.exp_id))
print(args.database)
print(args.model)
print('batch size:',args.batch_size)

device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

Info = h5py.File(datainfo, 'r')  # index, ref_ids
index = Info['index']
index = index[:, args.exp_id % index.shape[1]]  # np.random.permutation(N)
ref_ids = Info['ref_ids'][0, :]  #
# max_len = int(Info['max_len'][0])
max_len = args.frame_num
trainindex = index[0:int(np.ceil((1 - args.test_ratio - args.val_ratio) * len(index)))]
testindex = index[int(np.ceil((1 - args.test_ratio) * len(index))):len(index)]
train_index, val_index, test_index = [], [], []
for i in range(len(ref_ids)):
    train_index.append(i) if (ref_ids[i] in trainindex) else \
        test_index.append(i) if (ref_ids[i] in testindex) else \
            val_index.append(i)

scale = Info['scores'][0, :].max()  # label normalization factor
train_dataset = VQARDataset(features_dir, train_index, max_len, scale=scale)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
val_dataset = VQARDataset(features_dir, val_index, max_len, scale=scale)
val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=0)
if args.test_ratio > 0:
    test_dataset = VQARDataset(features_dir, test_index, max_len, scale=scale)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=0)
model = ADCMT(ori_dim=4096, dim=1024, width=args.width, depth=args.depth,dropout=args.dropout,emb_dropout=args.emb_dropout, num_frames=args.frame_num).cuda(args.deviceID) if torch.cuda.is_available() else ADCMT()
# print(model)
print('\n')

if not os.path.exists('weights'):
    os.makedirs('weights')
trained_model_file = 'weights/{}-{}-EXP{}'.format(args.model, args.database, args.exp_id)

if not args.disable_visualization:  # Tensorboard Visualization
    writer = SummaryWriter(log_dir='{}/EXP{}-{}-{}-{}-{}-{}-{}-94'
                           .format(args.log_dir, args.exp_id, args.database, args.model,
                                   args.lr, args.batch_size, args.epochs,
                                   datetime.datetime.now().strftime("%I-%M%p_%B-%d-%Y")))

criterion = nn.L1Loss()  # L1 loss
optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.decay_ratio)
best_val_criterion = -1  # SROCC min

c_sum = 0
for epoch in range(args.epochs):
    start_time1 = time.time()
    print('\n')
    print('-Epoch {}/{} Learning...'.format(epoch+1, args.epochs))
    # Train
    model.train()
    L = 0
    c = 0
    y_train = []
    y_pred = []
    trainL = list(train_loader)
    for i, (videos, lengths, scores, cls) in enumerate(trainL):
        # print('--Train Iter ', i)
        features = videos
        labels = scores
        for e in scores:
            y_train.append(e.item())
        features = features.to(device).float()
        labels = labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(features)
        for e in outputs[0]:
            y_pred.append(e.item())
        loss_s = criterion(outputs[0], labels)
        loss_c, cls_num = mini_CDCR_loss(outputs[1], labels, mode='MS', ranks=3, quantile=0.3, tau=args.tau)
        loss_c = loss_c.to(device)
        loss = (1-args.beta) * loss_s + args.beta * loss_c
        loss.backward()
        optimizer.step()
        L = L + loss.item()

    y_train = np.array(y_train)
    y_pred = np.array(y_pred)
    train_loss = L / (i + 1)
    train_PLCC = stats.pearsonr(y_pred, y_train)[0]
    train_SROCC = stats.spearmanr(y_pred, y_train)[0]
    train_RMSE = np.sqrt(((y_pred - y_train) ** 2).mean())
    train_KROCC = stats.kendalltau(y_pred, y_train)[0]
    print(' --Train loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}'.format(train_loss, train_SROCC, train_KROCC, train_PLCC, train_RMSE))
    args.beta = args.beta * 0.98

    model.eval()
    # Val
    y_val = []
    y_pred = []
    L = 0
    valL = list(val_loader)
    with torch.no_grad():
        for i, (videos, lengths, scores, cls) in enumerate(valL):
            #             print('--Val Iter ', i)
            features = videos
            labels = scores
            for e in scores:
                y_val.append(e.item())
            features = features.to(device).float()
            labels = labels.to(device).float()
            outputs = model(features)
            for e in outputs[0]:
                y_pred.append(e.item())
            loss_s = criterion(outputs[0], labels)
            loss = (1 - args.beta) * loss_s
            L = L + loss.item()
    y_val = np.array(y_val)
    y_pred = np.array(y_pred)
    val_loss = L / (i + 1)
    val_PLCC = stats.pearsonr(y_pred, y_val)[0]
    val_SROCC = stats.spearmanr(y_pred, y_val)[0]
    val_RMSE = np.sqrt(((y_pred - y_val) ** 2).mean())
    val_KROCC = kendalltau(y_pred, y_val)[0]
    print(" --Val loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
          .format(val_loss, val_SROCC, val_KROCC, val_PLCC, val_RMSE))

    # Test
    if not args.notest_during_training:
        y_test = []
        y_pred = []
        L = 0
        testL = list(test_loader)
        with torch.no_grad():
            for i, (videos, _, scores, cls) in enumerate(testL):
                #                 print('--Test Iter ', i)
                features = videos
                labels = scores
                for e in scores:
                    y_test.append(e.item())
                features = features.to(device).float()
                labels = labels.to(device).float()
                outputs = model(features)
                for e in outputs[0]:
                    y_pred.append(e.item())
                loss_s = criterion(outputs[0], labels)
                loss = (1 - args.beta) * loss_s
                L = L + loss.item()
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        test_loss = L / (i + 1)
        PLCC = stats.pearsonr(y_pred, y_test)[0]
        SROCC = stats.spearmanr(y_pred, y_test)[0]
        RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())
        KROCC = kendalltau(y_pred, y_test)[0]
        print(" --Test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
              .format(test_loss, SROCC, KROCC, PLCC, RMSE))

    if not args.disable_visualization:  # record training curves
        writer.add_scalar("loss/train", train_loss, epoch)  #
        writer.add_scalar("loss/val", val_loss, epoch)  #
        writer.add_scalar("SROCC/val", val_SROCC, epoch)  #
        writer.add_scalar("KROCC/val", val_KROCC, epoch)  #
        writer.add_scalar("PLCC/val", val_PLCC, epoch)  #
        writer.add_scalar("RMSE/val", val_RMSE, epoch)  #
        if not args.notest_during_training:
            writer.add_scalar("loss/test", test_loss, epoch)  #
            writer.add_scalar("SROCC/test", SROCC, epoch)  #
            writer.add_scalar("KROCC/test", KROCC, epoch)  #
            writer.add_scalar("PLCC/test", PLCC, epoch)  #
            writer.add_scalar("RMSE/test", RMSE, epoch)  #

    # Update the model with the best val_SROCC
    if val_SROCC > best_val_criterion:
        print("--EXP ID={}: Update best model using best_val_criterion in epoch {} ! >>>>>>>>>>>>>>>>>>".format(args.exp_id, epoch+1))
        print(" -Val results: val loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
              .format(val_loss, val_SROCC, val_KROCC, val_PLCC, val_RMSE))
        if not args.notest_during_training:
            print(" -Test results: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                  .format(test_loss, SROCC, KROCC, PLCC, RMSE))
        best_epoch = epoch
        best_loss = test_loss
        best_SROCC = SROCC
        best_KROCC = KROCC
        best_PLCC = PLCC
        best_RMSE = RMSE
        target = y_test
        pred = y_pred
        torch.save(model.state_dict(), trained_model_file)
        best_val_criterion = val_SROCC  # update best val SROCC

    end_time1 = time.time()
    Ep_time = end_time1 - start_time1
    print('-Epoch {} takes {:.4f} s'.format(epoch+1, Ep_time))
# Final Test
# if args.batch_size == 0:
#     model.load_state_dict(torch.load(trained_model_file))  #
#     model.eval()
#     with torch.no_grad():
#         y_test = []
#         y_pred = []
#         L = 0
#         testL = list(test_loader)
#         for i, (videos, _, scores) in enumerate(testL):
#             features = videos
#             labels = scores
#             for e in scores:
#                 y_test.append(e.item())
#             features = features.to(device).float()
#             labels = labels.to(device).float()
#             outputs = model(features)
#             for e in outputs:
#                 y_pred.append(e.item())
#             loss = criterion(outputs, labels)
#             L = L + loss.item()
#     y_test = np.array(y_test)
#     y_pred = np.array(y_pred)
#     test_loss = L / (i + 1)
#     PLCC = stats.pearsonr(y_pred, y_test)[0]
#     SROCC = stats.spearmanr(y_pred, y_test)[0]
#     RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())
#     KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
#     print('\n')
#     print("Final Test results in epoch {}: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
#           .format(best_epoch+1, test_loss, SROCC, KROCC, PLCC, RMSE))
print('\n')
print("Final test results in epoch {}: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
      .format(best_epoch+1, best_loss, best_SROCC, best_KROCC, best_PLCC, best_RMSE))
print(target)
print('----------------------------------')
print(pred)
