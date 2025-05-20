import numpy as np


def get_features(i, epoch, features, labels):
    # features [B, T+2, dim]
    if epoch == 0:
        features = features.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        print('获取所有批次32个样本的所有特征...')
        np.save('visualize/reg_CVQR-e{}-{}.npy'.format(epoch, i), features)
        np.save('visualize/l-{}.npy'.format(i), labels)
    elif (epoch+1)%10 == 0:
        features = features.detach().cpu().numpy()
        print('获取epoch{}所有批次32个样本的所有特征...'.format(epoch+1))
        np.save('visualize/reg_CVQR-e{}-{}.npy'.format(epoch, i), features)
