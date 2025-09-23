'''
@Time : 2024/11/12 9:53
@Auth : Qizhi Li
'''
import torch
import torch.nn as nn


class EntropyRegularization(nn.Module):

    def __init__(self, lambda_param=0.1):
        super().__init__()
        self.lambda_param = lambda_param

    def forward(self, inputs):
        entropy = -torch.sum(inputs * torch.log(inputs + 1e-8), dim=1)
        return entropy.mean()


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, lambda_c=1.0):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.lambda_c = lambda_c

    def forward(self, features, labels):
        # 计算中心损失
        batch_size = features.size(0)
        distmat = torch.pow(features.unsqueeze(1) - self.centers, 2)
        distmat = torch.sum(distmat, dim=2)

        # 获取对应的类别中心
        labels = labels.view(-1, 1)
        loss = torch.mean(distmat.gather(1, labels).squeeze())

        return self.lambda_c * loss