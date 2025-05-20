from torchviz import make_dot
import torch
from models.ADCMT import CVQT

import os
os.environ["PATH"] += os.pathsep + 'D:/Graphviz/bin/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample = torch.randn([32, 240, 4096]).to(device)
model = CVQT(depth=1, width=3, dim=1024).to(device)

x = sample.requires_grad_(True)  # 定义一个网络的输入值
y = model(x)  # 获取网络的预测值

MyConvNetVis = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
MyConvNetVis.format = "png"
# 指定文件生成的文件夹
MyConvNetVis.directory = "Network_Architecture"
# 生成文件
MyConvNetVis.view()