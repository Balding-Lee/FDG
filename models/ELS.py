'''
@Time : 2025/3/28 20:39
@Auth : Qizhi Li
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW


def cross_entropy_loss(pred_class_logits, gt_classes, eps=0.3, alpha=0.2, reduction='none'):
    # num_classes = pred_class_logits.size(1)

    # if eps >= 0:
    #     smooth_param = eps
    # else:
    #     # Adaptive label smooth regularization
    #     soft_label = F.softmax(pred_class_logits, dim=1)
    #     smooth_param = alpha * soft_label[torch.arange(soft_label.size(0)), gt_classes].unsqueeze(1)
    #
    # log_probs = F.log_softmax(pred_class_logits, dim=1)
    # with torch.no_grad():
    #     targets = torch.ones_like(log_probs)
    #     targets *= smooth_param / (num_classes - 1)
    #     targets.scatter_(1, gt_classes.data.unsqueeze(1), (1 - smooth_param))
    #
    # loss = (-targets * log_probs).sum(dim=1)
    #
    # with torch.no_grad():
    #     non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)
    # if reduction is not None:
    #     loss = loss.sum() / non_zero_cnt

    if eps >= 0:
        # 固定平滑：标签1变为 (1 - eps)，标签0变为 eps
        smoothed_targets = gt_classes * (1 - eps) + (1 - gt_classes) * eps
    else:
        # 自适应平滑：先计算预测概率
        probs = torch.sigmoid(pred_class_logits)
        # 对正例，平滑程度与预测概率成比例；对负例同理
        adaptive_eps = alpha * (gt_classes * probs + (1 - gt_classes) * (1 - probs))
        smoothed_targets = gt_classes * (1 - adaptive_eps) + (1 - gt_classes) * adaptive_eps

    # 计算二分类的交叉熵损失（logits版本自动内部调用 sigmoid）
    loss = F.binary_cross_entropy_with_logits(pred_class_logits, smoothed_targets, reduction='none')

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


class DomainClassifier(nn.Module):
    def __init__(self, lr, in_size=768):
        super().__init__()
        self.dense = nn.Linear(in_size, in_size)
        self.dropout = nn.Dropout(0.0)
        self.out_proj = nn.Linear(in_size, 1)  # 2 domains
        self.optimizer = AdamW(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class DomainDiscriminators(nn.Module):
    def __init__(self, num_domains, lr, in_size=768):
        super().__init__()
        # 因为有个目标域所以要-1
        self.domain_num = num_domains - 1
        self.domain_class = nn.ModuleList([DomainClassifier(lr, in_size) for _ in
                                           range((self.domain_num - 1) * self.domain_num // 2)])

    def forward(self, features, global_step=None, total_step=None):
        if global_step is not None and total_step is not None:
            progress = float(global_step) / float(total_step)
            lmda = 2 / (1 + math.exp(-5 * progress)) - 1
        else:
            lmda = 1.

        j_idx = 0
        loss_domain_disc_list_ = []
        error_domain_disc_list_ = []
        for i in range(self.domain_num):
            # 论文中提到的索引小的标签为0
            domain_t = torch.ones(features[i].size(0), requires_grad=False).type(torch.FloatTensor).to(
                features[0].device)
            for j in range(self.domain_num):
                # 论文中提到的索引大的标签为1
                if i < j:
                    domain_f = torch.zeros(features[j].size(0), requires_grad=False).type(torch.FloatTensor).to(
                        features[0].device)
                    logits_t = self.domain_class[j_idx](features[i].detach()).squeeze(1)
                    logits_f = self.domain_class[j_idx](features[j].detach()).squeeze(1)
                    error_domain_dis = ((1 - F.sigmoid(logits_t)).mean() + F.sigmoid(logits_f).mean()) * 0.5

                    # domain_discriminator_loss = ((
                    #     F.binary_cross_entropy_with_logits(logits_t, domain_t) +
                    #     F.binary_cross_entropy_with_logits(logits_f, domain_f)) * 0.5
                    # )
                    domain_discriminator_loss = ((
                            cross_entropy_loss(logits_t, domain_t, reduction='mean') +
                            cross_entropy_loss(logits_f, domain_f, reduction='mean')
                    ) * 0.5)

                    self.domain_class[j_idx].optimizer.zero_grad()
                    domain_discriminator_loss.backward()
                    self.domain_class[j_idx].optimizer.step()
                    j_idx += 1
                    error_domain_disc_list_.append(error_domain_dis.detach().item())
                    loss_domain_disc_list_.append(domain_discriminator_loss.detach().item())
        domdis_losses = []
        j_idx = 0
        for i in range(self.domain_num):
            for j in range(self.domain_num):
                if i < j:
                    domain_t = torch.ones(features[j].size(0), requires_grad=False).type(torch.FloatTensor).to(
                        features[0].device)
                    domain_f = torch.zeros(features[i].size(0), requires_grad=False).type(torch.FloatTensor).to(
                        features[0].device)

                    logits_t = self.domain_class[j_idx](features[i]).detach().squeeze(1)
                    logits_f = self.domain_class[j_idx](features[j]).detach().squeeze(1)

                    # domain_discriminator_loss = ((
                    #     F.binary_cross_entropy_with_logits(logits_t, domain_f) +
                    #     F.binary_cross_entropy_with_logits(logits_f, domain_t))
                    #         * 0.5
                    # )
                    domain_discriminator_loss = ((
                            cross_entropy_loss(logits_t, domain_f, reduction='mean') +
                            cross_entropy_loss(logits_f, domain_t, reduction='mean')
                    ) * 0.5)

                    domdis_losses.append(domain_discriminator_loss * lmda)
                    j_idx += 1

        return torch.stack(domdis_losses).mean(), error_domain_disc_list_, loss_domain_disc_list_
