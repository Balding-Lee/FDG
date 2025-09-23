'''
@Time : 2024/7/22 10:23
@Auth : Qizhi Li
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForMaskedLM, AutoModel


class CoVWeightingLoss(nn.Module):

    """
        Wrapper of the BaseLoss which weighs the losses to the Cov-Weighting method,
        where the statistics are maintained through Welford's algorithm. But now for 32 losses.
    """

    def __init__(self, mean_sort, mean_decay_param, device, num_losses=3, save_losses=False, target_domain=None):
        super().__init__()
        self.device = device
        self.save_losses = save_losses
        self.target_domain = target_domain
        self.num_losses = num_losses

        # How to compute the mean statistics: Full mean or decaying mean.
        self.mean_decay = True if mean_sort == 'decay' else False
        self.mean_decay_param = mean_decay_param

        self.current_iter = -1
        self.alphas = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)

        # Initialize all running statistics at 0.
        self.running_mean_L = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        self.running_mean_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        self.running_S_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        self.running_std_l = None

    def forward(self, unweighted_losses, losses_names=None, iteration=None):
        # Retrieve the unweighted losses.
        # unweighted_losses = super(CoVWeightingLoss, self).forward(pred, target)

        # Put the losses in a list. Just for computing the weights.
        L = torch.tensor(unweighted_losses, requires_grad=False).to(self.device)

        # If we are doing validation, we would like to return an unweighted loss be able
        # to see if we do not overfit on the training set.
        if not self.train:
            return torch.sum(L)

        # Increase the current iteration parameter.
        self.current_iter += 1
        # If we are at the zero-th iteration, set L0 to L. Else use the running mean.
        L0 = L.clone() if self.current_iter == 0 else self.running_mean_L
        # Compute the loss ratios for the current iteration given the current loss L.
        l = L / L0

        # If we are in the first iteration set alphas to all 1/32
        if self.current_iter <= 1:
            self.alphas = (torch.ones((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
                           / self.num_losses)
        # Else, apply the loss weighting method.
        else:
            ls = self.running_std_l / self.running_mean_l
            self.alphas = ls / torch.sum(ls)

        # Apply Welford's algorithm to keep running means, variances of L,l. But only do this throughout
        # training the model.
        # 1. Compute the decay parameter the computing the mean.
        if self.current_iter == 0:
            mean_param = 0.0
        elif self.current_iter > 0 and self.mean_decay:
            mean_param = self.mean_decay_param
        else:
            mean_param = (1. - 1 / (self.current_iter + 1))

        # 2. Update the statistics for l
        x_l = l.clone().detach()
        new_mean_l = mean_param * self.running_mean_l + (1 - mean_param) * x_l
        self.running_S_l += (x_l - self.running_mean_l) * (x_l - new_mean_l)
        self.running_mean_l = new_mean_l

        # The variance is S / (t - 1), but we have current_iter = t - 1
        running_variance_l = self.running_S_l / (self.current_iter + 1)
        self.running_std_l = torch.sqrt(running_variance_l + 1e-8)

        # 3. Update the statistics for L
        x_L = L.clone().detach()
        self.running_mean_L = mean_param * self.running_mean_L + (1 - mean_param) * x_L

        # Get the weighted losses and perform a standard back-pass.
        weighted_losses = [self.alphas[i] * unweighted_losses[i] for i in range(len(unweighted_losses))]

        if self.save_losses:
            assert iteration is not None
            assert len(weighted_losses) == len(losses_names)

            losses_curve_path = '../results/losses/losses_curve_{}.txt'.format(self.target_domain)
            weight_curve_path = '../results/losses/weight_curve_{}.txt'.format(self.target_domain)
            if os.path.exists(losses_curve_path):
                os.remove(losses_curve_path)

            if os.path.exists(weight_curve_path):
                os.remove(weight_curve_path)

            with open(losses_curve_path, 'a') as f:
                f.write('Iter: %d\t' % iteration)
                for i in range(len(weighted_losses)):
                    f.write('%s: %.4f\t' % (losses_names[i], weighted_losses[i]))
                f.write('\n')

            with open(weight_curve_path, 'a') as f:
                f.write('Iter: %d\t' % iteration)
                for i in range(len(unweighted_losses)):
                    f.write('%s: %f\t' % (losses_names[i], self.alphas[i]))

                f.write('\n')

        loss = sum(weighted_losses)
        return loss


class EntropyRegularization(nn.Module):

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.ce = nn.CrossEntropyLoss()
        self.lambda_param = 0.1

    def forward(self, inputs):
        probs = self.softmax(inputs)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean()

        return self.lambda_param * entropy


class FuzzyClassificationLoss(nn.Module):

    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.sigmoid = nn.Sigmoid()
        self.mse_loss = nn.MSELoss()

    def forward(self, logits, labels):
        fuzzy_membership_degree = self.sigmoid(logits)
        one_hot_labels = F.one_hot(labels, num_classes=self.num_labels).float()

        cls_loss = self.mse_loss(fuzzy_membership_degree, one_hot_labels)

        return cls_loss



class FuzzyMembershipSigmoid(nn.Module):

    def __init__(self, device, num_labels, mode='mse', lambda_reg=1.0, domain_label=0.5):
        super().__init__()
        self.mode = mode
        self.num_labels = num_labels
        self.device = device
        self.sigmoid = nn.Sigmoid()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_reg = lambda_reg
        self.domain_label = domain_label

    def forward(self, logits):
        membership_degree = self.sigmoid(logits)

        if self.mode == 'mse':
            membership_labels = torch.full([logits.shape[0], self.num_labels], self.domain_label).to(self.device)
            diff = membership_degree - membership_labels
            fm_loss = torch.mean(diff ** 2)

        return fm_loss


class FuzzyMembershipGaussian(nn.Module):

    def __init__(self, device, num_labels, mode='mse', lambda_reg=1.0):
        super().__init__()
        self.device = device
        self.mode = mode
        self.num_labels = num_labels
        self.center = nn.Parameter(torch.randn(num_labels), requires_grad=True)
        self.mu = nn.Parameter(torch.randn(num_labels), requires_grad=True)
        self.lambda_reg = lambda_reg

    def forward(self, logits, labels):
        """

        :param logits: [batch_size * num_domains, num_domains]
        :param labels: [batch_size * num_domains]
        :return:
        """
        # _, indices = labels.max(dim=1)

        # Extract the c and mu corresponding to different labels and expand one dimension for broadcasting
        # shape: [batch_size * num_domains, 1]
        center_selected = self.center[labels].unsqueeze(1)
        # shape: [batch_size * num_domains]
        mu_selected = self.mu[labels].unsqueeze(1)

        membership_degree = torch.exp(-((logits - center_selected) ** 2) / (2 * mu_selected ** 2))

        if self.mode == 'mse':
            membership_labels = torch.full(logits.shape, 0.5).to(self.device)
            diff = membership_degree - membership_labels
            fm_loss = torch.mean(diff ** 2)

        return fm_loss


class FuzzyContrastiveLearning(nn.Module):

    def __init__(self, device, sigma=1, num_classes=2):
        super().__init__()
        self.sigma = sigma
        self.device = device
        self.num_classes = num_classes
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, sentence_representation, labels):
        # Obtain fuzzy relationship matrix
        # shape: (batch_size * 3, 1)
        norm = sentence_representation.pow(2).sum(dim=1, keepdim=True)  # Calculate the square of each row
        # (a - b)^2 = a^2 + b^2 - 2ab
        distances = norm + norm.T - 2 * torch.matmul(sentence_representation, sentence_representation.T)

        # R = exp(-\frac{||x - y|}{\sigma})
        fuzzy_relation = torch.exp(-distances / (2 * self.sigma ** 2))

        one_hot = torch.eye(self.num_classes)[labels.cpu()]
        one_hot = one_hot.to(self.device)

        # shape: (batch_size * num_domains, batch_size * num_domains)
        label_similarity = torch.matmul(one_hot, one_hot.T)

        # Calculate the molecular: Sum the similarity of positive sample pairs
        # (batch_size, 1)
        positive_sum = torch.sum(fuzzy_relation * label_similarity, dim=1)

        # Calculate the denominator: Sum the similarities of all samples
        # (batch_size, 1)
        all_sum = torch.sum(fuzzy_relation, dim=1)

        # Compute contrastive loss (batch_size,)
        loss = -torch.log(positive_sum / (all_sum + 1e-8))

        loss = torch.mean(loss)

        return loss


class SentenceOrthogonality(nn.Module):
    def __init__(self, in_size, out_size, device, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.hidden_size = 768

        self.encode_linear = nn.Linear(in_size, out_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.mse_loss = nn.MSELoss()
        if num_classes == 2:
            self.ce_loss = nn.BCELoss()
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(0.1)

    def forward(self, sentence_representations, labels):
        outputs = self.sigmoid(self.encode_linear(sentence_representations))

        sentence_similarity = self.sigmoid(torch.matmul(outputs, outputs.T))

        # Determine the number of unique values in the vector
        num_classes = self.num_classes
        # num_classes = len(torch.unique(labels))

        # Create a identity matrix and use index operations
        # to map the elements in the vector to the rows of the identity matrix
        # shape: (batch_size * num_domains, num_classes)
        one_hot = torch.eye(num_classes)[labels.cpu()]
        one_hot = one_hot.to(self.device)

        # shape: (batch_size * num_domains, batch_size * num_domains)
        label_similarity = torch.matmul(one_hot, one_hot.T)

        # label_similarity = torch.LongTensor(label_similarity).to(self.device)
        contrastive_loss = self.ce_loss(sentence_similarity, label_similarity)

        return contrastive_loss


class ClassificationHeadModel(nn.Module):

    def __init__(self, model_path, s_num_labels=2, d_num_labels=3):
        super().__init__()
        self.model_config = AutoConfig.from_pretrained(model_path)
        if 'base' in model_path:
            hidden_size = 768
        else:
            hidden_size = 1024
        self.model = AutoModel.from_pretrained(model_path, config=self.model_config)
        self.sentence_classification_head = nn.Linear(hidden_size, s_num_labels)
        self.domain_classification_head = nn.Linear(hidden_size, d_num_labels)

    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.model(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        s_logits = self.sentence_classification_head(last_hidden_state)
        d_logits = self.domain_classification_head(last_hidden_state)

        return s_logits, d_logits, last_hidden_state
