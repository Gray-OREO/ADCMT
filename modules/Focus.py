import torch
from torch import nn
from torchvision import models
from einops import rearrange


def focus_video_build(video, indices, focus_rate):  # video[b,c,t,h,w], indices[b,t]
    # 对一批视频根据输入参数实现focus操作
    video = rearrange(video, 'b c t h w -> b t c h w')
    focus_res = int(video.shape[1] * focus_rate)
    indices = indices.detach().cpu().numpy().tolist()
    focus_video = torch.zeros([video.shape[0],focus_res,video.shape[2],video.shape[3],video.shape[4]]).cuda(1)
    for b in range(video.shape[0]):
        indices[b] = indices[b][0:focus_res]
        indices[b].sort()
        for t in range(focus_res):
            T = indices[b].pop(0)
            focus_video[b,t] = video[b,T]
    return rearrange(focus_video, 'b t c h w -> b c t h w')


class Focus(nn.Module):
    '''
    根据视频前部分帧，送入预训练网络获得时序重要性先验，减少后续送入transformer的数据维度
    '''
    def __init__(self, seq_len):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4096, seq_len, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x[b,c,t,h,w]
        feature = get_features(x, device='cuda:1')  # [b,4096]
        indices = self.mlp(feature)
        if x.shape[0] == 1:
            indices = indices.unsqueeze(0)  # batch_size设置为1时仍会报错？
        # print(indices.shape)
        indices = torch.sort(indices, 1, descending=True).indices
        res = focus_video_build(x, indices, focus_rate=0.5).cuda(1)
        return res


def get_features(video_data, device='cuda'):
    """feature extraction"""
    extractor = ResNet50().to(device)
    extractor.eval()
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    b = 0
    video_data = video_data.reshape([-1, 120, 3, 540, 960])
    with torch.no_grad():
        while b <= video_data.shape[0]-1:
            batch = video_data[b, 0].unsqueeze(0).to(device)
            features_mean, features_std = extractor(batch)
            output1 = torch.cat((output1, features_mean), 0)
            output2 = torch.cat((output2, features_std), 0)
            output = torch.cat((output1, output2), 1).squeeze()
            b += 1
    return output


class ResNet50(torch.nn.Module):
    """Modified ResNet50 for feature extraction"""

    def __init__(self):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        # features@: 7->res5c
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


if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    x = torch.rand(1, 3, 120, 540, 960).to(device)
    # sample = torch.zeros(240, 540, 960)
    # y = torch.rand(2, 3, 540, 960).to(device)
    model = Focus(seq_len=120).to(device)
    out = model(x)
    print(out.shape)
