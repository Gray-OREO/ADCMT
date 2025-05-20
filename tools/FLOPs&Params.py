import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
# from ViViT import ViViTBackbone
from models.Temporal_T import ViViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViViT(1, 240).to(device)
input = torch.randn(2, 240, 4096)
macs, params = profile(model, inputs=(input,))
print('FLOPs = ', macs/10**9, 'G')
print('params = ', params/10**6, 'M')