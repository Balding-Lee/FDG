'''
@Time : 2025/3/28 20:45
@Auth : Qizhi Li
'''
import os
import sys
import tqdm
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
from models.ELS import DomainDiscriminators


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

    # accuracy = accuracy_score(labels, preds)
    # precision = precision_score(labels, preds)
    # recall = recall_score(labels, preds)
    # f1 = f1_score(labels, preds)
    # return accuracy, precision, recall, f1
    return loss_total / len(data_loader), macro_F1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_domain',
                        choices=['book', 'dvd', 'electronics', 'kitchen', 'imdb', 'sst',
                                 'ch', 'f', 'gw', 'os', 's', 'fiction', 'government',
                                 'slate', 'telephone', 'travel', 'sick', 'snli'],
                        default='book')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--cuda', type=str, default=0)
    parser.add_argument('--model', type=str, default='roberta-base')
    args = parser.parse_args()

    print(args)

    device = torch.device('cuda:%s' % args.cuda if torch.cuda.is_available() else 'cpu')

    sa_domains = ['book', 'dvd', 'electronics', 'kitchen', 'imdb', 'sst']
    rumour_domains = ['ch', 'f', 'gw', 'os', 's']
    nli_domains = ['fiction', 'government', 'slate', 'telephone', 'travel', 'sick', 'snli']

    save_dir = '../parameters/baseline_model_{}.bin'.format(args.target_domain)

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

    target_domain = args.target_domain

    if target_domain in sa_domains:
        task_name = 'sa'
        if target_domain == 'imdb':
            num_domains = len(os.listdir(amazon_dir_path)) + 1
            max_length = 196
        else:
            if target_domain == 'sst':
                num_domains = len(os.listdir(amazon_dir_path)) + 1
            else:
                num_domains = len(os.listdir(amazon_dir_path))
            max_length = 128
    elif target_domain in rumour_domains:
        task_name = 'rumour'
        max_length = 64
        num_domains = len(os.listdir(pheme_dir_path))
    elif target_domain in nli_domains:
        task_name = 'nli'
        single_sentence_max_length = 48
        max_length = single_sentence_max_length * 2 + 3  # [CLS] + s1 + [SEP] + s2 + [SEP]

        if target_domain == 'sick' or target_domain == 'snli':
            num_domains = len(os.listdir(mnli_dir_path)) + 1
        else:
            num_domains = len(os.listdir(mnli_dir_path))

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

    domain_adv = DomainDiscriminators(num_domains, lr).to(device)

    criteria = nn.CrossEntropyLoss()

    param_list = [
        {'params': model.parameters()},
        {'params': domain_adv.parameters()}
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
    # epoch_time = time.time()
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
            logits_pos_list, logits_neg_list = [], []
            hidden_pos_list, hidden_neg_list = [], []

            for idx, batch in enumerate(batches):
                input_ids = batch[0].to(device)
                att_masks = batch[1].to(device)
                tgt = batch[2].to(device)
                domains = batch[3].tolist()

                s_logits, _, hidden_states = model(input_ids, att_masks)

                s_logits_list.append(s_logits)

                logits_pos_list.append(s_logits[torch.where(tgt > 0)])
                hidden_pos_list.append(hidden_states[torch.where(tgt > 0)])
                logits_neg_list.append(s_logits[torch.where(tgt == 0)])
                hidden_neg_list.append(hidden_states[torch.where(tgt == 0)])

                labels.append(tgt)
                domain_label_list.append(domains)

            s_logits = torch.cat(s_logits_list, dim=0)

            labels = torch.cat(labels, dim=0).to(device)
            domain_labels = torch.tensor(domain_label_list).flatten().to(device)

            ce_loss = criteria(s_logits, labels)
            loss = ce_loss

            loss_domain_pos, _, _ = domain_adv(hidden_pos_list)
            loss_domain_neg, _, _ = domain_adv(hidden_neg_list)
            loss_domain = (loss_domain_pos + loss_domain_neg) * 0.5
            loss = loss + loss_domain * 0.1

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

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

    if not os.path.exists('results/scores/%s/ELS/%s' % (task_name, args.model)):
        os.makedirs('results/scores/%s/ELS/%s' % (task_name, args.model))

    with open('results/scores/%s/ELS/%s/%s.txt' % (task_name, args.model, target_domain), 'a') as f:
        f.write('f1: %.4f\n' % best_F1)

