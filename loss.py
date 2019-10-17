import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CULoss(nn.Module):
    def __init__(self):
        super(CULoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, true_conf_batch, lamba, labeled_pred_conf_batch, unlabeled_pred_conf_batch, ltrue_label_batch, utrue_label_batch):
        obj1 = torch.mean((true_conf_batch - lamba) * self.loss(labeled_pred_conf_batch, torch.squeeze(ltrue_label_batch, dim=1).long()))
        obj2 = lamba[0] * torch.mean(self.loss(unlabeled_pred_conf_batch, torch.squeeze(utrue_label_batch, dim=1).long()))
        objective = obj1 + obj2
        return objective


class KDLoss(nn.Module):
    def __init__(self):
        super(KDLoss, self).__init__()
        self.sup_loss = nn.CrossEntropyLoss()
        self.ts_loss = nn.MSELoss()

    def forward(self, upred_label_batch, utrue_label_batch, lpred_conf_batch, ly_prior, lteacher_feature_batch, lstudent_feature_batch):
        obj1 = torch.mean(self.sup_loss(upred_label_batch, torch.squeeze(utrue_label_batch, dim=1).long()))
        obj2 = torch.mean(lpred_conf_batch / ly_prior * self.ts_loss(lteacher_feature_batch, lstudent_feature_batch))
        objective = obj1 + obj2
        return objective


class NoConfLoss(nn.Module):
    def __init__(self):
        super(NoConfLoss, self).__init__()
        self.sup_loss = nn.CrossEntropyLoss()
        self.ts_loss = nn.MSELoss()

    def forward(self, alpha, pred_label_batch, true_label_batch, teacher_feature_batch, student_feature_batch):
        obj1 = torch.mean(self.sup_loss(pred_label_batch, torch.squeeze(true_label_batch, dim=1).long()))
        obj2 = alpha * torch.mean(self.ts_loss(teacher_feature_batch, student_feature_batch))
        objective = obj1 + obj2
        return objective
