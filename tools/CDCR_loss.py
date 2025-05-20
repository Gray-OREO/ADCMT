import numpy as np
import torch
from torch import einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from tools.Score2Rank import KM_indices_for_levels, MS_indices_for_levels


def QDC(mb1):
    B = mb1.shape[0]
    x = repeat(mb1, 'b 1 -> b B', B=B)
    x = x - x.T
    x = torch.maximum(x, -x)
    return 1 / (x + 1)


def mini_CDCR_loss(CDCRs, labels, mode='KM', ranks=3, quantile=0.1, tau=0.5):
    assert mode in {'KM', 'MS'}, 'clustering mode must be either KM (K-Means) or MS (Mean-Shift)'
    B = CDCRs.shape[0]
    # CVQRs [batch, length, embed_dim]
    if CDCRs.dim() == 3:
        CDCRs = CDCRs.mean(dim = 1)
    alpha = QDC(labels)
    # alpha = 1
    if mode == 'KM':
        if B < ranks:
            ranks = B-1
        RankIndices = KM_indices_for_levels(labels.detach().cpu(), rank=ranks)
    else:
        RankIndices = MS_indices_for_levels(labels.detach().cpu(), quantile=quantile)
    # CVQRs [batch, embed_dim]
    CDCRs = F.normalize(CDCRs, p=2, dim=1)
    sim = torch.mm(CDCRs, CDCRs.T) / tau * alpha  # scaled sim matrix
    sim = torch.exp(sim)  # e^sim
    sim = sim - torch.diag(torch.diag(sim))  # 0-diag

    Loss_cvqr = 0
    for b in range(B):
        positive_indices = []
        for i in range(len(RankIndices)):
            if b in RankIndices[i]:
                positive_indices = RankIndices[i]
        num_positive = len(positive_indices)
        denominator = torch.sum(sim[b])
        numerator = denominator
        # Avoid for inf value when no positive samples
        if num_positive != 1:
            numerator = 0
            for j in positive_indices:
                numerator += sim[b, j]
        l = - torch.log(numerator/denominator)
        Loss_cvqr += l / num_positive
    return Loss_cvqr/B, len(RankIndices)


def CoarseClassifier(labels):
    labels = labels.detach().cpu().numpy().astype(int)
    # mcls = list(set(labels))
    mcls = set(labels)
    mcls_dict = {}
    for cls in mcls:
        # print(cls)
        tmp_dict = {cls: []}
        mcls_dict.update(tmp_dict)
    i = 0
    for c in labels:
        res = mcls_dict[c]
        res.append(i)
        tmp_dict = {labels[i]: res}
        mcls_dict.update(tmp_dict)
        i += 1
    return mcls_dict


def mini_ContraLoss(samples, labels, tau=0.5):
    B = samples.shape[0]
    # CDCRs [batch, embed_dim]

    samples = F.normalize(samples, p=2, dim=1)
    sim = torch.mm(samples, samples.T) / tau  # sim matrix
    # print(type(sim))
    sim = torch.exp(sim)  # e^sim
    sim = sim - torch.diag(torch.diag(sim))  # 0-diag

    cls_dict = CoarseClassifier(labels)
    RankIndices = list(cls_dict.values())
    Loss_c = 0
    for b in range(B):
        positive_indices = []
        for i in range(len(RankIndices)):
            if b in RankIndices[i]:
                positive_indices = RankIndices[i]
        num_positive = len(positive_indices)
        denominator = torch.sum(sim[b])
        numerator = denominator
        # Avoid for inf value when no positive samples
        if num_positive != 1:
            numerator = 0
            for j in positive_indices:
                numerator += sim[b, j]
        l = - torch.log(numerator / denominator)
        Loss_c += l / num_positive
    return Loss_c / B


if __name__ == '__main__':
    QRs = torch.randn([32, 240, 4096])
    labels = torch.randn([32,1])
    l = mini_CDCR_loss(QRs, labels, ranks=5)
    print(l)
    # diag = torch.diag(output) * torch.eye(2)
    # print(diag)