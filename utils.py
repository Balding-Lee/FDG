'''
@Time : 2024/7/22 10:20
@Auth : Qizhi Li
'''
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, data, targets, tokenizer, max_length, domains=None):
        self.data = data
        self.targets = targets
        self.domains = domains
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        target = int(self.targets[index])

        if self.domains is not None:
            domain = self.domains[index]

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 转换为一维向量
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        if self.domains is None:
            return input_ids, attention_mask, target
        else:
            return input_ids, attention_mask, target, domain


def read_imdb_csv(dir_path, template):
    data = pd.read_csv(dir_path, header=0)
    original_texts = data['sentence'].tolist()
    labels = data['labels'].tolist()

    texts = []
    for text in original_texts:
        texts.append(template + text)

    return texts, labels


def load_imdb_dataset(dir_path, tokenizer, max_length, template):
    val_texts, val_labels = read_imdb_csv(os.path.join(dir_path, 'dev.csv'), template)
    test_texts, test_labels = read_imdb_csv(os.path.join(dir_path, 'test.csv'), template)

    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

    return val_dataset, test_dataset


def read_sst_tsv(dir_path, template):
    data = pd.read_csv(dir_path, sep='\t', header=0)
    original_texts = data['sentence'].tolist()
    labels = data['label'].tolist()

    texts = []
    for text in original_texts:
        texts.append(template + text)

    return texts, labels


def load_sst_dataset(dir_path, tokenizer, max_length, template):
    val_texts, val_labels = read_sst_tsv(os.path.join(dir_path, 'dev.tsv'), template)
    test_texts, test_labels = read_sst_tsv(os.path.join(dir_path, 'test.tsv'), template)

    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

    return val_dataset, test_dataset


def load_amazon_dataset(dir_path, target_domain, tokenizer, max_length, template):
    train_texts, train_labels, train_domains, train_datasets, train_domain_labels = [], [], [], [], []
    test_texts, test_labels = [], []
    d_verbalizer = {}

    domains = os.listdir(dir_path)

    i = 0  # domain 的标签
    for domain in domains:
        if domain != target_domain:
            with open(os.path.join(dir_path, domain, 'all_data.txt'), 'r') as f:
                texts = f.readlines()

            train_domains.append(len(texts))
            for text in texts:
                line = text.strip().split(' ||| ')
                if len(line) == 2:
                    train_texts.append(template + line[0])
                    train_labels.append(line[1])
                    train_domain_labels.append(i)
            d_verbalizer[domain] = i
            i += 1

    for num_samples in train_domains:
        train_dataset = CustomDataset(train_texts[:num_samples], train_labels[:num_samples],
                                      tokenizer, max_length, train_domain_labels[:num_samples])
        del train_texts[:num_samples]
        del train_labels[:num_samples]
        del train_domain_labels[:num_samples]
        train_datasets.append(train_dataset)

    if target_domain in domains:
        with open(os.path.join(dir_path, target_domain, 'test.txt'), 'r') as f:
            lines = f.readlines()

        for text in lines:
            line = text.strip().split(' ||| ')
            if len(line) == 2:
                test_texts.append(template + line[0])
                test_labels.append(line[1])

        test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

        return train_datasets, test_dataset, d_verbalizer
    else:
        return train_datasets, d_verbalizer


def load_sst_dataset_original(dir_path, tokenizer, max_length):
    val_texts, val_labels = read_sst_tsv_original(os.path.join(dir_path, 'dev.tsv'))
    test_texts, test_labels = read_sst_tsv_original(os.path.join(dir_path, 'test.tsv'))

    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

    return val_dataset, test_dataset


def load_imdb_dataset_original(dir_path, tokenizer, max_length):
    val_texts, val_labels = read_imdb_csv_original(os.path.join(dir_path, 'dev.csv'))
    test_texts, test_labels = read_imdb_csv_original(os.path.join(dir_path, 'test.csv'))

    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

    return val_dataset, test_dataset


def read_sst_tsv_original(dir_path):
    data = pd.read_csv(dir_path, sep='\t', header=0)
    texts = data['sentence'].tolist()
    labels = data['label'].tolist()

    return texts, labels


def read_imdb_csv_original(dir_path):
    data = pd.read_csv(dir_path, header=0)
    texts = data['sentence'].tolist()
    labels = data['labels'].tolist()

    return texts, labels


def load_amazon_dataset_raw_texts(dir_path, target_domain):
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []
    domains = os.listdir(dir_path)

    for domain in domains:

        if domain != target_domain:
            train_texts.append([])
            train_labels.append([])
            with open(os.path.join(dir_path, domain, 'all_data.txt'), 'r') as f:
                texts = f.readlines()

            for text in texts:
                line = text.strip().split(' ||| ')
                if len(line) == 2:
                    train_texts[-1].append(line[0])
                    train_labels[-1].append(int(line[1]))

    if target_domain in domains:
        with open(os.path.join(dir_path, target_domain, 'test.txt'), 'r') as f:
            lines = f.readlines()

        for text in lines:
            line = text.strip().split(' ||| ')
            if len(line) == 2:
                test_texts.append(line[0])
                test_labels.append(int(line[1]))

    return train_texts, train_labels, test_texts, test_labels


def load_amazon_dataset_original(dir_path, target_domain, tokenizer, max_length, is_name=False):
    train_texts, train_labels, train_domains, train_datasets, train_domain_labels = [], [], [], [], []
    test_texts, test_labels = [], []

    domains = os.listdir(dir_path)

    source_domain_names = []

    i = 0  # domain 的标签
    for domain in domains:
        if domain != target_domain:
            source_domain_names.append(domain)
            with open(os.path.join(dir_path, domain, 'all_data.txt'), 'r') as f:
                texts = f.readlines()

            train_domains.append(len(texts))
            for text in texts:
                line = text.strip().split(' ||| ')
                if len(line) == 2:
                    train_texts.append(line[0])
                    train_labels.append(line[1])
                    train_domain_labels.append(i)
            i += 1

    for num_samples in train_domains:
        train_dataset = CustomDataset(train_texts[:num_samples], train_labels[:num_samples],
                                      tokenizer, max_length, train_domain_labels[:num_samples])
        del train_texts[:num_samples]
        del train_labels[:num_samples]
        del train_domain_labels[:num_samples]
        train_datasets.append(train_dataset)

    if target_domain in domains:
        with open(os.path.join(dir_path, target_domain, 'test.txt'), 'r') as f:
            lines = f.readlines()

        for text in lines:
            line = text.strip().split(' ||| ')
            if len(line) == 2:
                test_texts.append(line[0])
                test_labels.append(line[1])

        test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

        if is_name:
            return train_datasets, test_dataset, i, source_domain_names
        else:
            return train_datasets, test_dataset, i
    else:
        if is_name:
            return train_datasets, i, source_domain_names
        else:
            return train_datasets, i


def read_pheme_csv(dir_path, i=None):
    data = pd.read_csv(dir_path, header=0)
    original_texts = data['texts'].tolist()
    labels = data['labels'].tolist()
    domain_labels = []

    texts = []
    for text in original_texts:
        texts.append(text)
        if i is not None:
            domain_labels.append(i)

    if i is not None:
        return texts, labels, domain_labels
    else:
        return texts, labels


def load_pheme_dataset(dir_path, target_domain, tokenizer, max_length):
    train_texts, train_labels, train_datasets = [], [], []

    domains = os.listdir(dir_path)
    domain_mapping = {
        'ch': 'charliehebdo',
        'f': 'ferguson',
        'gw': 'germanwings',
        'os': 'ottawashooting',
        's': 'sydneysiege',
    }

    i = 0  # domain 的标签
    for domain in domains:
        if domain != '{}.csv'.format(domain_mapping[target_domain]):
            texts, labels, train_domain_labels = read_pheme_csv(os.path.join(dir_path, domain), i)
            train_dataset = CustomDataset(texts, labels, tokenizer, max_length, train_domain_labels)
            train_datasets.append(train_dataset)
            i += 1

    texts, labels = read_pheme_csv(os.path.join(dir_path, '{}.csv'.format(domain_mapping[target_domain])))
    test_dataset = CustomDataset(texts, labels, tokenizer, max_length)

    return train_datasets, test_dataset, i


def read_nli_csv(dir_path, tokenizer, single_sentence_max_length, i=None):
    df = pd.read_csv(dir_path, header=0)
    sentence1 = df['sentence1'].tolist()
    sentence2 = df['sentence2'].tolist()
    labels = df['label'].tolist()
    domain_labels = []

    texts = []
    for index in range(len(sentence1)):
        # 如果文本过长, 将文本截断, 同时去除收尾[CLS] [SEP]
        sentence1_truncation = tokenizer.decode(tokenizer.encode(sentence1[index],
                                                                 truncation=True,
                                                                 max_length=single_sentence_max_length)[1: -1])
        sentence2_truncation = tokenizer.decode(tokenizer.encode(sentence2[index],
                                                                 truncation=True,
                                                                 max_length=single_sentence_max_length)[1: -1])

        text = sentence1_truncation + tokenizer.sep_token + sentence2_truncation
        texts.append(text)
        if i is not None:
            domain_labels.append(i)

    if i is not None:
        return texts, labels, domain_labels
    else:
        return texts, labels


def load_mnli_datasets(dir_path, target_domain, tokenizer, single_sentence_max_length, max_length):
    train_datasets, train_domain_labels = [], []
    domains = os.listdir(dir_path)

    i = 0  # domain 的标签
    for domain in domains:
        if domain != target_domain:
            texts, labels, domain_labels = read_nli_csv(os.path.join(dir_path, domain, 'train.csv'),
                                                        tokenizer, single_sentence_max_length, i)
            train_dataset = CustomDataset(texts, labels, tokenizer, max_length, domain_labels)
            train_datasets.append(train_dataset)

            i += 1

    if target_domain in domains:
        test_texts, test_labels = read_nli_csv(os.path.join(dir_path, target_domain, 'test.csv'),
                                               tokenizer, single_sentence_max_length)

        test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

        return train_datasets, test_dataset, i
    else:
        return train_datasets, i


def load_snli_sick_datasets(dir_path, tokenizer, single_sentence_max_length, max_length):
    val_texts, val_labels = read_nli_csv(os.path.join(dir_path, 'dev.csv'),
                                         tokenizer, single_sentence_max_length)
    test_texts, test_labels = read_nli_csv(os.path.join(dir_path, 'test.csv'),
                                           tokenizer, single_sentence_max_length)

    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

    return val_dataset, test_dataset


class ANet(nn.Module):
    def __init__(self, in_feature):
        super(ANet, self).__init__()
        self.layer = nn.Linear(in_feature, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x)
        x = self.sigmoid(x)
        return x


def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct


def calculate(source_feature: torch.Tensor, target_feature: torch.Tensor,
              device, progress=True, training_epochs=10):
    """
    Calculate the :math:`/mathcal{A}`-distance, which is a measure for distribution discrepancy.

    The definition is :math:`dist_/mathcal{A} = 2 (1-2/epsilon)`, where :math:`/epsilon` is the
    test error of a classifier trained to discriminate the source from the target.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        device (torch.device)
        progress (bool): if True, displays a the progress of training A-Net
        training_epochs (int): the number of epochs when training the classifier

    Returns:
        :math:`/mathcal{A}`-distance
    """
    source_label = torch.ones((source_feature.shape[0], 1))
    target_label = torch.zeros((target_feature.shape[0], 1))
    feature = torch.cat([source_feature, target_feature], dim=0)
    label = torch.cat([source_label, target_label], dim=0)

    dataset = TensorDataset(feature, label)
    length = len(dataset)
    train_size = int(0.8 * length)
    val_size = length - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

    anet = ANet(feature.shape[1]).to(device)
    optimizer = SGD(anet.parameters(), lr=0.01)
    # a_distance = 2.0
    best_acc = float('inf')
    out_acc = 0.0
    for epoch in range(training_epochs):
        anet.train()
        for (x, label) in train_loader:
            x = x.to(device)
            label = label.to(device)
            anet.zero_grad()
            y = anet(x)
            loss = F.binary_cross_entropy(y, label)
            loss.backward()
            optimizer.step()

        anet.eval()
        accs = []
        with torch.no_grad():
            for (x, label) in val_loader:
                x = x.to(device)
                label = label.to(device)
                y = anet(x)
                acc = binary_accuracy(y, label)
                # meter.update(acc, x.shape[0])
                accs.append(acc.detach().cpu().numpy())

        mean_acc = np.mean(np.array(accs))
        # error = 1 - meter.avg / 100
        # a_distance = 2 * (1 - 2 * error)
        if abs(mean_acc - 50) < best_acc:
            best_acc = abs(mean_acc - 50)
            if mean_acc > 50:
                out_acc = mean_acc
            else:
                out_acc = 50 + abs(mean_acc - 50)
        if progress:
            # print("epoch {} accuracy: {}".format(epoch, meter.avg))
            print("epoch {} accuracy: {} out acc: {}".format(epoch, mean_acc, out_acc))

    error = 1 - out_acc / 100
    a_distance = 2 * (1 - 2 * error)

    return a_distance


def compute_mmd(x, y, sigma=1.0):
    """
    计算MMD (Maximum Mean Discrepancy)
    :param x: tensor, shape (batch_size, feature_dim), 目标分布的样本
    :param y: tensor, shape (batch_size, feature_dim), 源分布的样本
    :param sigma: float, 高斯核的带宽参数
    :return: MMD的值
    """

    # 计算高斯核
    def rbf_kernel(x, y, sigma):
        x_norm = torch.sum(x ** 2, dim=1).view(-1, 1)  # shape: (batch_size, 1)
        y_norm = torch.sum(y ** 2, dim=1).view(1, -1)  # shape: (1, batch_size)
        dist = x_norm + y_norm - 2 * torch.mm(x, y.t())  # shape: (batch_size, batch_size)
        return torch.exp(-dist / (2 * sigma ** 2))  # 高斯核计算

    # 计算MMD
    xx_kernel = rbf_kernel(x, x, sigma)  # 同域之间的内积
    yy_kernel = rbf_kernel(y, y, sigma)  # 同域之间的内积
    xy_kernel = rbf_kernel(x, y, sigma)  # 异域之间的内积

    # 计算MMD
    mmd = torch.mean(xx_kernel) + torch.mean(yy_kernel) - 2 * torch.mean(xy_kernel)

    return mmd


