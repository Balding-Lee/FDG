'''
@Time : 2024/11/5 10:28
@Auth : Qizhi Li
'''
import os
import sys
import tqdm
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import RobertaTokenizer, RobertaConfig, RobertaForMaskedLM
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

sys.path.append('..')
import utils
from models import FuzzyDG


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def preparing_features(model, dataloader, desc, args):
    pos_features, neg_features = [], []
    features = []
    for batch in tqdm.tqdm(dataloader, desc=desc):
        input_ids = batch[0].to(device)
        att_masks = batch[1].to(device)
        labels = batch[2]

        # feature = model(input_ids, att_masks)
        # shape: [batch_size, 768]
        _, _, feature = model(input_ids, att_masks)

        pos_features.append(feature[labels == 1])
        neg_features.append(feature[labels == 0])
        features.append(feature)

        # if len(features) >= 125:
        #     break

    features = torch.cat(features, dim=0)
    pos_features = torch.cat(pos_features, dim=0)
    neg_features = torch.cat(neg_features, dim=0)
    # print(features.shape)
    return features, pos_features, neg_features
    # return pos_features, neg_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scl', action='store_true', help='using sentence contrastive learning')
    parser.add_argument('--contrastive', choices=['original', 'fuzzy'], default='fuzzy')
    parser.add_argument('--fm', choices=['gaussian', 'sigmoid', 'none'], default='sigmoid')
    parser.add_argument('--target_domain', choices=['book', 'dvd', 'electronics', 'kitchen', 'imdb', 'sst'],
                        default='book')
    parser.add_argument('--tsne', action='store_true', help='draw t-sne fig')
    parser.add_argument('--pad', action='store_true', help='calculate PAD')
    parser.add_argument('--mmd', action='store_true', help='calculate MMD')
    args = parser.parse_args()

    print(args)

    seed = 9
    set_seed(seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_path = '/home/liqizhi/workspace/KernelWord/pretrained_parameters/roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_path)

    batch_size = 16
    epochs = 1

    if args.target_domain == 'imdb':
        max_length = 196
    else:
        max_length = 128

    s_verbalizer = {
        'good': 1,
        'bad': 0
    }

    target_domain = args.target_domain

    amazon_dir_path = '../datasets/amazon'
    imdb_dir_path = '../datasets/imdb'
    sst_dir_path = '../datasets/sst'

    if args.target_domain == 'sst' or args.target_domain == 'imdb':
        source_datasets, _, source_domain_names = utils.load_amazon_dataset_original(
            amazon_dir_path,
            target_domain,
            tokenizer,
            max_length,
            is_name=True
        )
        _, target_dataset = utils.load_sst_dataset_original(sst_dir_path, tokenizer, max_length)
    else:
        source_datasets, target_dataset, _, source_domain_names = utils.load_amazon_dataset_original(
            amazon_dir_path,
            target_domain,
            tokenizer,
            max_length,
            is_name=True
        )

    source_dataloader = [DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=12)
                         for train_dataset in source_datasets]
    target_dataloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=12)

    if target_domain == 'imdb' or target_domain == 'sst':
        model = FuzzyDG.ClassificationHeadModel(model_path, s_num_labels=2, d_num_labels=4).to(device)
    else:
        model = FuzzyDG.ClassificationHeadModel(model_path, s_num_labels=2, d_num_labels=3).to(device)

    for param in model.parameters():
        param.requires_grad = False

    save_dir = '../parameters/model_{}_{}_contrastive_{}_{}.bin'.format(target_domain, args.fm, args.scl, args.contrastive)

    model.load_state_dict(torch.load(save_dir), strict=False)

    with torch.no_grad():
        model.eval()
        i = 1
        source_pos_features, source_neg_features = [], []
        source_features = []
        for source in source_dataloader:
            # shape: (2000, 768)
            features, pos_features, neg_features = preparing_features(model, source,
                                                                      desc='preparing the features of the {}-th domain'.format(
                                                                          i),
                                                                      args=args)
            source_pos_features.append(pos_features)
            source_neg_features.append(neg_features)
            source_features.append(features)
            i += 1

        target_features, target_pos_features, target_neg_features = preparing_features(model, target_dataloader,
                                                                                       desc='preparing the target domain features',
                                                                                       args=args)

    if args.pad:
        pADs = []
        for i in range(len(source_features)):
            print('The {}-th source features'.format(i))
            pAD = utils.calculate(source_features[i], target_features, device)
            pADs.append(pAD)

        print('seed %d, pAD: %.4f' % (seed, (sum(pADs) / len(pADs))))

        with open('../results/pad_{}_ours.txt'.format(args.target_domain), 'a') as f:
            text = 'seed: %d\tpAD: %.4f\n' % (seed, (sum(pADs) / len(pADs)))
            f.write(text)

    if args.mmd:
        mmds = []
        # target_features = target_features.detach().cpu().numpy()
        for i in range(len(source_features)):
            # source_feature = source_features[i].detach().cpu().numpy()
            mmd = utils.compute_mmd(target_features, source_features[i], sigma=1.0)
            mmds.append(mmd)
            print('The {}-th domain: {:.4f}'.format(i, mmd))

        print('mean MMD: %.4f' % (sum(mmds) / len(mmds)))

    # ======================= t-SNE =======================
    if args.tsne:
        if args.target_domain == 'sst' or args.target_domain == 'imdb':
            domain_names = [args.target_domain]
            domain_names.extend(source_domain_names)
        else:
            domain_names = source_domain_names
            domain_names.append(args.target_domain)
        domain_name_id_mapping, domain_id_name_mapping = {}, {}
        i = 0
        for domain_name in domain_names:
            domain_name_id_mapping[domain_name] = i
            domain_id_name_mapping[i] = domain_name
            i += 1

        label2name = {
            0: 'negative',
            1: 'positive'
        }

        if args.target_domain == 'sst' or args.target_domain == 'imdb':
            pos_features = [target_pos_features]
            pos_features.extend(source_pos_features)
        else:
            pos_features = source_pos_features
            pos_features.append(target_pos_features)
        pos_features_np = np.vstack([tensor.cpu().numpy() for tensor in pos_features])
        pos_labels = np.array([[i, 1] for i, features in enumerate(pos_features) for _ in range(len(features))])

        if args.target_domain == 'sst' or args.target_domain == 'imdb':
            neg_features = [target_neg_features]
            neg_features.extend(source_neg_features)
        else:
            neg_features = source_neg_features
            neg_features.append(target_neg_features)
        neg_features_np = np.vstack([tensor.cpu().numpy() for tensor in neg_features])
        neg_labels = np.array([[i, 0] for i, features in enumerate(neg_features) for _ in range(len(features))])

        # 定义颜色和形状
        if args.target_domain == 'sst' or args.target_domain == 'imdb':
            colors = ['#F48C8C', '#F3F1AD', '#EAFFD0', '#95E1D3', '#F4D8A5']
            edgecolors = ['#f38181', '#FCE38A', '#C0F0D2', '#87CDC0', '#F4BF9D']
        else:
            colors = ['#F48C8C', '#F3F1AD', '#EAFFD0', '#95E1D3']
            edgecolors = ['#f38181', '#FCE38A', '#C0F0D2', '#87CDC0']

        tsne = TSNE(n_components=2,
                    random_state=seed)
        # data_2d = tsne.fit_transform(data)
        pos_data_2d = tsne.fit_transform(pos_features_np)
        neg_data_2d = tsne.fit_transform(neg_features_np)

        plt.figure(figsize=(10, 10))

        # 定义颜色和形状
        if args.target_domain == 'sst' or args.target_domain == 'imdb':
            colors = ['#F48C8C', '#F3F1AD', '#EAFFD0', '#95E1D3', '#F4D8A5']
            edgecolors = ['#f38181', '#FCE38A', '#C0F0D2', '#87CDC0', '#F4BF9D']
        else:
            colors = ['#F48C8C', '#F3F1AD', '#EAFFD0', '#95E1D3']
            edgecolors = ['#f38181', '#FCE38A', '#C0F0D2', '#87CDC0']
        # shapes = ['o', '^']

        for domain_id in domain_id_name_mapping:
            idx = (pos_labels[:, 0] == domain_id)
            plt.scatter(pos_data_2d[idx, 0], pos_data_2d[idx, 1], c=colors[domain_id], s=30,
                        edgecolor=edgecolors[domain_id], label=f'Domain: {domain_id_name_mapping[domain_id]}')

        plt.xticks([])
        plt.yticks([])

        plt.legend(fontsize=18, framealpha=0.5)
        plt.savefig('../results/visualize/t-sne target {}-pos-scl {}-fm {}.pdf'.format(args.target_domain,
                                                                                       args.contrastive,
                                                                                       args.fm))
        # plt.show()

        plt.cla()

        for domain_id in domain_id_name_mapping:
            idx = (neg_labels[:, 0] == domain_id)
            plt.scatter(neg_data_2d[idx, 0], neg_data_2d[idx, 1], c=colors[domain_id], s=30,
                        edgecolor=edgecolors[domain_id], label=f'Domain: {domain_id_name_mapping[domain_id]}')

        plt.xticks([])
        plt.yticks([])

        plt.legend(fontsize=18, framealpha=0.5)
        plt.savefig('../results/visualize/t-sne target {}-neg-scl {}-fm {}.pdf'.format(args.target_domain,
                                                                                       args.contrastive,
                                                                                       args.fm))