import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# from module import Attention, PreNorm, FeedForward
import numpy as np


def mask(Input, head_num, interval, mode='no-periodic'):
    # Input [B, length(heads+sequence), dim]
    v_mask = torch.zeros([Input.shape[1]], device='cuda')
        # .to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    v_mask[0:head_num] = 1
    if interval > 3 and mode == 'periodic':
        interval = interval % 4 + 1
    for i in range(2, len(v_mask), interval):
        v_mask[i] = 1
    v_mask = v_mask.unsqueeze(0).unsqueeze(-1)
    res = torch.mul(Input, v_mask)
    return res
    # return Input


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer_block(nn.Module):
    def __init__(self, dim, width, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.width = width
        self.dim = dim
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)

        for _ in range(width):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    
    def forward(self, x):      
        CVQRs_MB = torch.zeros([self.width, x.shape[0], self.dim], device='cuda')  # [width,B,dim]
            # .to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        i = 0
        h = x
        recali_x = torch.mul(x, x[:, 1].unsqueeze(1))
        #recali_x = torch.cat((x[:, :2], recali_x[:, 2:]), dim=1)
        # recali_x = x
        for attn, ff in self.layers:
            x = mask(recali_x, 2, i+1)
            x = attn(x) + h
            x = ff(x) + x
            CVQRs_MB[i] = x[:, 1]
            i += 1
        CVQRs_MB = rearrange(CVQRs_MB, 'w b d -> b w d')
        return self.norm(x), CVQRs_MB


class ADCMT(nn.Module):
    def __init__(self, dim, num_classes=1, num_frames=240,
                 depth=1, width=3, heads=8, pool='cls', dropout=0.,
                 emb_dropout=0., scale_dim=4, ori_dim=4096):
        super().__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        dim_head = dim // heads

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames+2, dim))  # changed delete

        self.quality_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cvqr_token = nn.Parameter(torch.randn(1, 1, dim))

        # self.temporal_transformer = Transformer_block(dim, depth, heads, dim_head, dim*scale_dim, dropout)
        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            self.blocks.append(Transformer_block(dim, width, heads, dim_head, dim * scale_dim, dropout))

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool
        self.width = width
        self.depth = depth

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes, bias=False)
        )
        self.proj = nn.Linear(ori_dim, dim, bias=False)  # dim_red
        # self.recali = nn.Parameter(torch.ones(1, 1, dim))
        self.agg = nn.Conv2d(depth, 1, kernel_size=(1, 1), bias=False)

    def forward(self, x):
        #  x[B,240,4096]
        # x = Rearrange('b c t h w -> b t c h w')(x)
        x = self.proj(x)  # dim_red
        b, t, d = x.shape

        reg_tokens = repeat(self.quality_token, '() n d -> b n d', b = b)
        CVQR_tokens = repeat(self.cvqr_token, '() n d -> b n d', b=b)
        x = torch.cat((reg_tokens, CVQR_tokens, x), dim=1)  # [B,242,4096]
        x += self.pos_embedding  # changed x += self.pos_embedding[:,:,:(n+1)]
        x = self.dropout(x)

        # x = rearrange(x, 'b t n d -> (b t) n d')
        CVQRs_heads = torch.zeros([self.depth, b, self.width, d], device='cuda')
            # .to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # [depth, B, width, dim]

        # for d in range(self.depth):
        #     x, CVQRs_head = self.temporal_transformer(x)  # [B, 242, dim], [B, width, dim]
        #     CVQRs_heads[d] = CVQRs_head
        D = 0
        for CVQT_block in self.blocks:
            x, CVQRs_head = CVQT_block(x)
            CVQRs_heads[D] = CVQRs_head        
            D +=1

        # h = x #v
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        CVQRs_heads = rearrange(CVQRs_heads, 'D B w d -> B D w d')
        return self.mlp_head(x), self.agg(CVQRs_heads).squeeze(1).squeeze(1)
    

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    # img = torch.ones([1, 16, 3, 224, 224]).cuda()
    sample = torch.randn([32, 240, 4096]).to(device)
    
    model = ADCMT(depth=2, width=4, dim=1024).to(device)
    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    # parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    # print('Trainable Parameters: %.3fM' % parameters)
    out = model(sample)[1]
    print(model)      # [B, num_classes]
    # ME_VMB = torch.zeros([self.width, x.shape[0], x.shape[1], self.dim]) # V numpy.save("visualize/ME{}.npy".format(i), x.detach().cpu())
