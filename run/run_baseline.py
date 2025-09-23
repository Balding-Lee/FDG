'''
@Time : 2024/11/5 16:42
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
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
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


def evaluate(model, data_loader, tgt_domain):
    preds = []
    labels = []
    loss_total = 0.0
    model.eval()

    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc='test on {}'.format(tgt_domain)):
            input_ids = batch[0].to(device)
            att_masks = batch[1].to(device)
            tgt = batch[2].to(device)
            labels.extend(tgt.cpu().tolist())

            logits, _, _ = model(input_ids, att_masks)

            pred = torch.argmax(logits, dim=-1)
            loss = F.cross_entropy(logits, tgt)
            loss_total += loss
            preds.extend(pred.cpu().detach().tolist())

    model.train()
    macro_F1 = f1_score(labels, preds, average='macro')

    return loss_total / len(data_loader), macro_F1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_domain',
                        choices=['book', 'dvd', 'electronics', 'kitchen', 'imdb', 'sst',
                                 'ch', 'f', 'gw', 'os', 's', 'fiction', 'government',
                                 'state', 'telephone', 'travel', 'sick', 'snli'],
                        default='book')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--cuda', type=str, default=0)
    parser.add_argument('--model', type=str, default='roberta-base')
    args = parser.parse_args()

    print(args)

    sa_domains = ['book', 'dvd', 'electronics', 'kitchen', 'imdb', 'sst']
    rumour_domains = ['ch', 'f', 'gw', 'os', 's']
    nli_domains = ['fiction', 'government', 'state', 'telephone', 'travel', 'sick', 'snli']

    device = torch.device('cuda:%s' % args.cuda if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # save_dir = '../parameters/model_{}_{}_contrastive_{}.bin'.format(args.target_domain, args.fm, args.scl)
    save_dir = '../parameters/baseline_model_{}.bin'.format(args.target_domain)

    target_domain = args.target_domain
    seed = args.seed
    set_seed(seed)

    model_path = '/sdc1/liqizhi/huggingface/%s' % args.model
    tokenizer = RobertaTokenizer.from_pretrained(model_path)

    amazon_dir_path = 'datasets/amazon'
    imdb_dir_path = 'datasets/imdb'
    sst_dir_path = 'datasets/sst'
    pheme_dir_path = 'datasets/PHEME'
    mnli_dir_path = 'datasets/NLI/MNLI'
    sick_dir_path = 'datasets/NLI/SICK'
    snli_dir_path = 'datasets/NLI/SNLI'

    # target_domain = args.target_domain

    s_verbalizer = {
        'good': 1,
        'bad': 0
    }

    if target_domain in sa_domains:
        task_name = 'sa'
        if target_domain == 'imdb':
            max_length = 196
        else:
            max_length = 128
    elif target_domain in rumour_domains:
        task_name = 'rumour'
        max_length = 64
    elif target_domain in nli_domains:
        task_name = 'nli'
        single_sentence_max_length = 48
        max_length = single_sentence_max_length * 2 + 3  # [CLS] + s1 + [SEP] + s2 + [SEP]

    batch_size = 16

    weight_decay = 1e-2
    lr = args.lr  # 0.000015
    num_epochs = 30

    if target_domain in sa_domains:
        if target_domain != 'sst' and target_domain != 'imdb':
            train_datasets, test_dataset, num_domain_labels = utils.load_amazon_dataset_original(
                amazon_dir_path,
                target_domain,
                tokenizer,
                max_length)
        else:
            train_datasets, num_domain_labels = utils.load_amazon_dataset_original(amazon_dir_path,
                                                                                   target_domain,
                                                                                   tokenizer, max_length)

            if args.target_domain == 'imdb':
                val_dataset, test_dataset = utils.load_imdb_dataset_original(imdb_dir_path, tokenizer,
                                                                             max_length)
            else:
                val_dataset, test_dataset = utils.load_sst_dataset_original(sst_dir_path, tokenizer, max_length)

            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12)
    elif target_domain in rumour_domains:
        train_datasets, test_dataset, num_domain_labels = utils.load_pheme_dataset(pheme_dir_path,
                                                                                   target_domain,
                                                                                   tokenizer, max_length)
    elif target_domain in nli_domains:
        if target_domain == 'sick':
            train_datasets, num_domain_labels = utils.load_mnli_datasets(mnli_dir_path, target_domain,
                                                                         tokenizer,
                                                                         single_sentence_max_length,
                                                                         max_length)
            val_dataset, test_dataset = utils.load_snli_sick_datasets(sick_dir_path, tokenizer,
                                                                      single_sentence_max_length,
                                                                      max_length)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12)
        elif target_domain == 'snli':
            train_datasets, num_domain_labels = utils.load_mnli_datasets(mnli_dir_path, target_domain,
                                                                         tokenizer,
                                                                         single_sentence_max_length,
                                                                         max_length)
            val_dataset, test_dataset = utils.load_snli_sick_datasets(snli_dir_path, tokenizer,
                                                                      single_sentence_max_length,
                                                                      max_length)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12)
        else:
            train_datasets, test_dataset, num_domain_labels = utils.load_mnli_datasets(mnli_dir_path,
                                                                                       target_domain,
                                                                                       tokenizer,
                                                                                       single_sentence_max_length,
                                                                                       max_length)

    if target_domain in sa_domains:
        train_dataloader = [DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
                            for train_dataset in train_datasets]
    else:
        train_dataloader = [DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12,
                                       drop_last=True) for train_dataset in train_datasets]

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=12)

    # model = FuzzyDG.PromptModel(model_path, tokenizer, s_verbalizer_ids, d_verbalizer_ids).to(device)
    if target_domain in nli_domains:
        if target_domain == 'sick' or target_domain == 'snli':
            model = FuzzyDG.ClassificationHeadModel(model_path, s_num_labels=3, d_num_labels=5).to(device)
        else:
            model = FuzzyDG.ClassificationHeadModel(model_path, s_num_labels=3, d_num_labels=4).to(device)
    elif target_domain in sa_domains:
        if target_domain == 'imdb' or target_domain == 'sst':
            model = FuzzyDG.ClassificationHeadModel(model_path, s_num_labels=2, d_num_labels=4).to(device)
        else:
            model = FuzzyDG.ClassificationHeadModel(model_path, s_num_labels=2, d_num_labels=3).to(device)
    else:
        model = FuzzyDG.ClassificationHeadModel(model_path, s_num_labels=2, d_num_labels=4).to(device)

    criteria = nn.CrossEntropyLoss()

    param_list = [
        {'params': model.parameters()},
    ]

    optimizer = optim.AdamW(
        param_list,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1.0e-8
    )

    best_F1 = 0.0
    best_val_loss = float('inf')
    epoch_time = time.time()
    batch_index = 0
    epoch_index = 0
    last_improve = 0
    require_improvement = 5
    flag = False
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_train_steps = 0

        # start_time = time.time()
        # train_iters = zip(*train_dataloader)
        pbar = tqdm.tqdm(zip(*train_dataloader),
                         desc='Seed {}, target domain {}, Training epoch {}'.format(seed, target_domain, epoch))
        for step, batches in enumerate(pbar):
            model.train()

            s_logits_list, d_logits_list, cls_hidden_states_list, labels, domain_label_list = [], [], [], [], []

            # 这里能够保证每次添加的数据都是不同域的数据,
            # 因为 train_dataloader 中遍历的就是不同域的数据
            for idx, batch in enumerate(batches):
                input_ids = batch[0].to(device)
                att_masks = batch[1].to(device)
                tgt = batch[2].to(device)
                domains = batch[3].tolist()

                s_logits, _, _ = model(input_ids, att_masks)

                s_logits_list.append(s_logits)

                labels.append(tgt)
                domain_label_list.append(domains)

            s_logits = torch.cat(s_logits_list, dim=0)

            labels = torch.cat(labels, dim=0).to(device)
            domain_labels = torch.tensor(domain_label_list).flatten().to(device)


            ce_loss = criteria(s_logits, labels)
            # er_loss = er(d_logits, domain_labels)

            # loss = cov_weighting_loss([ce_loss, er_loss, contrastive_loss])
            loss = ce_loss

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
        print('epoch duration: %.4f' % (time.time() - epoch_time))

        if target_domain == 'imdb' or target_domain == 'sst' or target_domain == 'sick' or target_domain == 'snli':
            val_loss, f1 = evaluate(model, val_dataloader, target_domain)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                _, test_f1 = evaluate(model, test_dataloader, target_domain)

                if test_f1 > best_F1:
                    best_F1 = test_f1

                last_improve = epoch_index

            print("Epoch {}, loss: {:.4f}, best F1: {:.4f}".format(epoch, val_loss, best_F1))

        else:
            _, f1 = evaluate(model, test_dataloader, target_domain)

            if f1 > best_F1:
                best_F1 = f1
                last_improve = epoch_index

            print("Epoch {}, F1: {:.4f}, ".format(epoch, f1))

    print('Best F1: {:.4f}'.format(best_F1))

    if not os.path.exists('results/scores/%s/CLS/%s' % (task_name, args.model)):
        os.makedirs('results/scores/%s/CLS/%s' % (task_name, args.model))

    with open('results/scores/%s/CLS/%s/%s.txt' % (task_name, args.model, target_domain), 'a') as f:
        f.write('f1: %.4f\n' % best_F1)
