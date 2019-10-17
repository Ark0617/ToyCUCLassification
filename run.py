import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import *
from dataset import ToyDataset
from loss import *
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
tb_writer = SummaryWriter()
parser = argparse.ArgumentParser(description='a toy version CU classification')

parser.add_argument('--lamba', type=float, default=0.01)
parser.add_argument('--predictor-hidden-dim', type=int, default=10)
parser.add_argument('--teacher-hidden-dim', type=int, default=10)
parser.add_argument('--student-hidden-dim', type=int, default=5)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=5000)
parser.add_argument('--teacher-pretrain-epochs', type=int, default=5000)
parser.add_argument('--predictor-epochs', type=int, default=5000)
parser.add_argument('--labeled-prior', type=float, default=0.3)
parser.add_argument('--data-num', type=int, default=7680)
parser.add_argument('--pred-lr', type=float, default=1e-1)
parser.add_argument('--teacher-lr', type=float, default=1e-1)
parser.add_argument('--student-lr', type=float, default=1e-1)
parser.add_argument('--eval_batch_size', type=int, default=64)
parser.add_argument('--use-conf', action='store_true')
parser.add_argument('--alpha', type=float, default=1)
args = parser.parse_args()

teacher_net = ToyTeacherNet(args.teacher_hidden_dim).to(device)
student_net = ToyStudentNet(args.student_hidden_dim).to(device)
conf_predictor = ToyConfPredictor(args.teacher_hidden_dim, args.predictor_hidden_dim).to(device)
student_feature_transform = nn.Linear(args.student_hidden_dim, args.teacher_hidden_dim)
teacher_optimizer = optim.Adam(teacher_net.parameters(), args.teacher_lr)
student_optimizer = optim.Adam(student_net.parameters(), args.student_lr)
predictor_optimizer = optim.Adam(conf_predictor.parameters(), args.pred_lr)
sup_teacher_criterion = nn.CrossEntropyLoss()
cu_criterion = CULoss()
kd_criterion = KDLoss()
noconf_criterion = NoConfLoss()
dataset = ToyDataset(args.data_num)
data = dataset.data
label = dataset.label
idx = np.random.permutation(args.data_num)
data = data[idx]
label = label[idx]

def evaluate_accuracy(net, data_batch, true_label):
    data_batch = torch.Tensor(data_batch).to(device)
    true_label = torch.Tensor(true_label).to(device)
    pred_label_vec = net(data_batch)
    pred_label = torch.argmax(pred_label_vec, dim=1).long()
    correct = 0
    true_label = torch.squeeze(true_label, dim=0).long()
    for i in range(pred_label.size(0)):
        if pred_label[i] == true_label[i]:
            correct += 1
    acc = correct / len(data_batch)
    return acc


def plot_decision_boundary(net, data, title_str):
    X = data #.numpy()
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = np.argmax(net(torch.Tensor(np.c_[xx.ravel(), yy.ravel()])).detach().numpy(), axis=1)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.6, cmap=plt.cm.Spectral)  # cmap=plt.cm.Spectral
    dataset.visualize_data()
    plt.show()


# ====Teacher pretraining====
for i in range(args.teacher_pretrain_epochs):
    id_batch = np.random.choice(args.data_num, args.batch_size)
    data_batch = torch.Tensor(data[id_batch]).to(device)
    label_batch = torch.squeeze(torch.Tensor(label[id_batch]), dim=1).to(device)
    pred_vec = teacher_net(Variable(data_batch))
    teacher_sup_loss = sup_teacher_criterion(pred_vec, label_batch.long())
    teacher_optimizer.zero_grad()
    teacher_sup_loss.backward()
    teacher_optimizer.step()
    if i % 100 == 0:
        print('Teacher Pretrain Epoch: {}, Teacher loss: {}'.format(i, teacher_sup_loss))
    tb_writer.add_scalar('teacher_pretrain_sup_loss', teacher_sup_loss, i)

teacher_acc = evaluate_accuracy(teacher_net, data, label)
print("teacher final accuracy: {}".format(teacher_acc))
#plot_decision_boundary(teacher_net, data, "Toy experiment Teacher classfification")

if args.use_conf:
# ====Use pretrained Teacher to give confidence scores=====

    conf_threshold = int(len(data) * args.labeled_prior)
    conf_data = torch.Tensor(data[:conf_threshold]).to(device)
    conf_label = torch.Tensor(label[:conf_threshold]).to(device)
    unconf_data = torch.Tensor(data[conf_threshold:]).to(device)
    unconf_label = torch.Tensor(label[conf_threshold:]).to(device)
    conf_vec = teacher_net(conf_data)
    conf = torch.empty(conf_data.size(0))
    for i in range(len(conf_vec)):
        conf[i] = conf_vec[i, conf_label[i].long()]


    # ====Train g(x): the predictor====
    for i in range(args.predictor_epochs):
        conf_id_batch = np.random.choice(len(conf_data), args.batch_size)
        conf_data_batch = conf_data[conf_id_batch]
        conf_label_batch = conf_label[conf_id_batch]
        conf_teacher_feature_batch = teacher_net.feature_extract(conf_data_batch)
        conf_z_i = torch.cat([conf_data_batch, conf_label_batch, conf_teacher_feature_batch], dim=1)
        conf_g_z = conf_predictor(conf_z_i)
        true_conf_batch = conf[conf_id_batch].to(device)
        # true_conf_batch, lamba, labeled_pred_conf_batch, unlabeled_pred_conf_batch, ltrue_label_batch, utrue_label_batch
        unconf_id_batch = np.random.choice(len(unconf_data), args.batch_size)
        unconf_data_batch = unconf_data[unconf_id_batch]
        unconf_label_batch = unconf_label[unconf_id_batch]
        unconf_teacher_feature_batch = teacher_net.feature_extract(unconf_data_batch)
        unconf_z_i = torch.cat([unconf_data_batch, unconf_label_batch, unconf_teacher_feature_batch], dim=1)
        unconf_g_z = conf_predictor(unconf_z_i)
        lamba = torch.Tensor(np.array([args.lamba] * true_conf_batch.shape[0])).to(device)
        cu_loss = cu_criterion(true_conf_batch, lamba, conf_g_z, unconf_g_z, conf_label_batch, unconf_label_batch)
        # print(cu_loss)
        predictor_optimizer.zero_grad()
        cu_loss.backward(retain_graph=True)
        predictor_optimizer.step()
        tb_writer.add_scalar('predictor_cu_loss', cu_loss, i)
        if i % 100 == 0:
            print('Predictor Training Epoch: {}, cu loss: {}'.format(i, cu_loss))

    # ====Use trained predictor to give confidence score to unconf data====

    unconf_z = torch.cat([unconf_data, unconf_label, teacher_net.feature_extract(unconf_data)], dim=1)
    pred_conf_vec = conf_predictor(unconf_z)
    pred_conf = torch.empty(pred_conf_vec.size(0))
    for i in range(len(unconf_z)):
        pred_conf[i] = pred_conf_vec[i, unconf_label[i].long()]

    total_conf = torch.cat([conf, pred_conf], dim=0)
    ly_prior = torch.empty(len(label))
    for i in range(len(label)):
        if label[i] == 0 or label[i] == 1:
            ly_prior[i] = 0.25
        else:
            ly_prior[i] = 0.5

    # ====Train Student====
    # upred_label_batch, utrue_label_batch, lpred_conf_batch, ly_prior, lteacher_feature_batch, lstudent_feature_batch
    for i in range(args.epochs):
        unconf_id_batch = np.random.choice(len(unconf_data), args.batch_size)
        unconf_data_batch = torch.Tensor(unconf_data[unconf_id_batch]).to(device)
        upred_label_batch = student_net(unconf_data_batch)
        utrue_label_batch = torch.Tensor(unconf_label[unconf_id_batch]).to(device)

        total_id_batch = np.random.choice(len(data), args.batch_size)
        total_data_batch = torch.Tensor(data[total_id_batch]).to(device)
        total_conf_batch = torch.Tensor(total_conf[total_id_batch]).to(device)
        ly_prior_batch = torch.Tensor(ly_prior[total_id_batch]).to(device)
        lteacher_feature_batch = teacher_net.feature_extract(total_data_batch)
        lstudent_feature_batch = student_feature_transform(student_net.feature_extract(total_data_batch))
        kd_loss = kd_criterion(upred_label_batch, utrue_label_batch, total_conf_batch, ly_prior_batch, lteacher_feature_batch, lstudent_feature_batch)
        student_optimizer.zero_grad()
        kd_loss.backward(retain_graph=True)
        student_optimizer.step()
        # evaluate...
        eval_id_batch = np.random.choice(len(data), args.eval_batch_size)
        eval_data_batch = torch.Tensor(data[eval_id_batch])
        eval_label_batch = torch.Tensor(label[eval_id_batch])
        acc = evaluate_accuracy(student_net, eval_data_batch, eval_label_batch)
        tb_writer.add_scalar('student_kd_loss', kd_loss, i)
        tb_writer.add_scalar('student_accuracy', acc, i)
        if i % 100 == 0:
            print('Student Training Epoch: {}, kd loss: {}, acc: {}'.format(i, kd_loss, acc))
    student_acc = evaluate_accuracy(student_net, data, label)
    print("student final accuracy: {}".format(student_acc))
    plot_decision_boundary(student_net, data, "Toy experiment Student classfification")
    # plt.show()
else:
    print('hhh')
    for i in range(args.epochs):
        id_batch = np.random.choice(len(data), args.batch_size)
        data_batch = torch.Tensor(data[id_batch]).to(device)
        true_label_batch = torch.Tensor(label[id_batch]).to(device)
        teacher_feature_batch = teacher_net.feature_extract(data_batch)
        student_feature_batch = student_feature_transform(student_net.feature_extract(data_batch))
        pred_label_batch = student_net(data_batch)
        noconf_loss = noconf_criterion(args.alpha, pred_label_batch, true_label_batch, teacher_feature_batch, student_feature_batch)
        student_optimizer.zero_grad()
        noconf_loss.backward(retain_graph=True)
        student_optimizer.step()
        eval_id_batch = np.random.choice(len(data), args.eval_batch_size)
        eval_data_batch = torch.Tensor(data[eval_id_batch])
        eval_label_batch = torch.Tensor(label[eval_id_batch])
        acc = evaluate_accuracy(student_net, eval_data_batch, eval_label_batch)
        tb_writer.add_scalar('student_loss_without_conf', noconf_loss, i)
        tb_writer.add_scalar('student_accuracy_without_conf', acc, i)
        if i % 100 == 0:
            print('Student Training Epoch: {},  loss: {}, acc: {}'.format(i, noconf_loss, acc))
    student_acc = evaluate_accuracy(student_net, data, label)
    print("student final accuracy: {}".format(student_acc))
    plot_decision_boundary(student_net, data, "Toy experiment Student classfification")
