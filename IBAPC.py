#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Linkun Fan
@Contact: flinkun@henu.edu.cn
@File: IBAPC.py
@Time: 2024/12/05 10:39 PM
"""


from __future__ import print_function
import os
import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import trimesh
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40
from model import PointNet, DGCNN
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from GFT import GFT, show_pl, GFT_opt
from torchviz import make_dot
import time
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")



def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

target_label = 2

def train(args, io):
    train_loader = DataLoader(
        ModelNet40(partition='train', num_points=args.num_points, test_poison_target=0, target_num=None,
                   model=None,
                   test_poison=False, GFT_noise = None), num_workers=0,
        batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points,test_poison_target=0, target_num = None, model=None, test_poison=False, GFT_noise = None), num_workers=0,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    test_poison_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points, test_poison_target=0, target_num = None, model=None, test_poison=True, GFT_noise = None), num_workers=0,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    path = "record_one"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)


    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    criterion = cal_loss
    best_test_acc = 0

    # initialization
    kk = 1
    poison_rate = 0.05  # pr
    fix_poison = True  # fp
    weight_benign = 0.5 # wb
    weight_attack = 0.5 # wa
    poison_weight_attack = 0.9 # pwa
    poison_weight_dis = 0.1 # pwd
    initial_noise_level = 0.1 # inl
    attack_lr = 0.01 # a_lr
    stage2_epoc = 0
    purt_low_begin = 0  # plb
    purt_high_begin = 0  # phb
    purt_low_size = 1024 # pls
    purt_high_size = 1024 # phs

    GFT_noise = np.random.uniform(low=-0.5, high=0.5, size=(args.num_points, 3)) * initial_noise_level

    ### 将不希望添加扰动的频率置零
    if not len(GFT_noise[0:purt_low_begin]) == 0:
        GFT_noise[0:purt_low_begin] = [0,0,0]
    if not len(GFT_noise[purt_low_begin+purt_low_size: purt_high_begin]) == 0:
        GFT_noise[purt_low_begin+purt_low_size: purt_high_begin] = [0,0,0]
    if not len(GFT_noise[purt_high_begin + purt_high_size: args.num_points]) == 0:
        GFT_noise[purt_high_begin + purt_high_size: args.num_points] = [0,0,0]

    GFT_noise = torch.tensor(GFT_noise).to(torch.device("cuda"))

    trigger_opt = optim.Adam([GFT_noise], lr=attack_lr, weight_decay=1e-4)
    scheduler_opt = CosineAnnealingLR(trigger_opt, args.epochs, eta_min=args.lr)

    benign_test_acc = []
    attack_test_acc = []
    indivadual_l2_loss = []
    plt_x = []

    train_poison_count = 0
    for epoch in range(args.epochs):
        plt_x.append(epoch)
        scheduler.step()
        scheduler_opt.step()
        train_loss = 0.0
        count = 0.0
        train_pred = []
        train_true = []

        attack_train_pred = []
        attack_train_true = []

        all_l2_loss = 0
        for data, label, v, conduct_poison in train_loader:
            v = v.to(device)
            label = torch.tensor(label, dtype=torch.int64)
            ori_label = label.clone()
            data, attack_label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            ####################################################################################################
            # Poison Train
            # The network parameters are unchanged, and the parameters of the poisoning function are updated to achieve two purposes:
            #1. The samples are classified as target categories after poisoning; 2. The deformation between the samples after poisoning and the clean samples is small
            ####################################################################################################
            model.eval()
            trigger_opt.zero_grad()
            GFT_noise.requires_grad_(True)
            GFT_noise.retain_grad()
            attack_label.fill_(target_label)
            data2 = GFT_opt(data.permute(0, 2, 1), GFT_noise, v)
            logits = model(data2.permute(0, 2, 1).float())
            attack_label = torch.tensor(attack_label, dtype=torch.int64)
            attack_loss = criterion(logits, attack_label)
            attack_preds = logits.max(dim=1)[1]

            attack_train_pred.append(attack_preds.detach().cpu().numpy())
            attack_train_true.append(attack_label.cpu().numpy())

            l2_loss = 0
            for i in range(len(data)):
                l2 = torch.norm(data[i].permute(1, 0) - data2[i], p=2)
                l2_loss += l2
            all_l2_loss += l2_loss

            all_loss = poison_weight_attack * attack_loss + poison_weight_dis * l2_loss
            all_loss.backward()
            trigger_opt.step()

            ### Zero the frequency at which you do not want to add a disturbance
            with torch.no_grad():
                if not len(GFT_noise[0:purt_low_begin]) == 0:
                    GFT_noise[0:purt_low_begin] = [0, 0, 0]
                if not len(GFT_noise[purt_low_begin + purt_low_size: purt_high_begin]) == 0:
                    GFT_noise[purt_low_begin + purt_low_size: purt_high_begin] = torch.tensor([0, 0, 0])
                if not len(GFT_noise[purt_high_begin + purt_high_size: args.num_points]) == 0:
                    GFT_noise[purt_high_begin + purt_high_size: args.num_points] = torch.tensor([0, 0, 0])

            if epoch % kk == 0:
                used_GFT_noise = GFT_noise.clone()

            ####################################################################################################
            # Benign Train
            # The parameters of the poison function remain unchanged, and the parameters of the network are updated to achieve two purposes:
            # 1. The samples without poison are correctly classified. 2 The poisoned samples are classified into the specified categories
            ####################################################################################################
            sss = np.where(conduct_poison == True)
            if not sss[0].size == 0:
                for kkkkk in sss:
                    for kkkk in kkkkk:
                        train_poison_count += 1
                        data[kkkk] = GFT_opt(data[kkkk].unsqueeze(0).permute(0, 2, 1), used_GFT_noise, v[kkkk].unsqueeze(0)).permute(0, 2, 1)
                        ori_label[kkkk] = torch.tensor([target_label])
            model.train()
            data, ori_label = data.to(device), ori_label.to(device).squeeze()

            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            benign_loss = criterion(logits, ori_label)

            loss = benign_loss
            loss.backward()
            opt.step()

            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(ori_label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        attack_train_true = np.concatenate(attack_train_true)
        attack_train_pred = np.concatenate(attack_train_pred)

        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)

        print('        indivadual l2_loss:{}'.format(all_l2_loss / len(train_true)))
        print('        train attack acc:{}'.format(metrics.accuracy_score(attack_train_true, attack_train_pred)))

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label, v, _ in test_loader:
            label = torch.tensor(label, dtype=torch.int64)
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)


        #test poison
        poison_test_acc = test_poison_opt(model, test_poison_loader, criterion, used_GFT_noise)
        indivadual_l2_loss.append((all_l2_loss / len(train_true)).detach().cpu().numpy())
        benign_test_acc.append(test_acc)
        attack_test_acc.append(poison_test_acc)

        plt.plot(plt_x, benign_test_acc, ls="-", lw=2, label="benign test accuracy")
        plt.plot(plt_x, attack_test_acc, ls="-", lw=2, label="attack success rate")

        plt.legend()
        plt.savefig(
            'record_one\\acc,plb{},phb{},pls{},phs{},a_lr{}, inl{}, fp{}, pr{}, wb{},wa{},pwa{},pwd{}.jpg'.format(purt_low_begin, purt_high_begin,purt_low_size,purt_high_size,attack_lr, initial_noise_level, fix_poison,poison_rate,
                weight_benign, weight_attack,
                poison_weight_attack,
                poison_weight_dis,
                ))

        plt.clf()

        plt.plot(plt_x, indivadual_l2_loss, ls="-", lw=2, label="indivadual_l2_loss")
        plt.legend()
        plt.savefig(
            'record_one\\L2 dis,plb{},phb{},pls{},phs{},a_lr{}, inl{}, fp{}, pr{}, wb{},wa{},pwa{},pwd{}.jpg'.format(purt_low_begin, purt_high_begin,purt_low_size,purt_high_size,attack_lr, initial_noise_level, fix_poison,poison_rate,
                weight_benign, weight_attack,
                poison_weight_attack,
                poison_weight_dis,
                ))

        plt.clf()

        if epoch %10 == 0 or epoch == 249:
            noise_path = 'record_one\\GFT_noise epoc{},plb{},phb{},pls{},phs{},a_lr{}, inl{}, fp{}, pr{}, wb{},wa{},pwa{},pwd{}'.format(epoch, purt_low_begin, purt_high_begin,purt_low_size,purt_high_size,attack_lr, initial_noise_level, fix_poison,poison_rate,
                weight_benign, weight_attack,
                poison_weight_attack,
                poison_weight_dis,
                    )
            if os.path.exists(noise_path):
                os.remove(noise_path)
            np.save(noise_path, used_GFT_noise.detach().cpu().numpy())
            torch.save(model.state_dict(), 'checkpoints/{}/models/epoc{} model.t7'.format(args.exp_name, epoch))
    sys.exit()


def test_poison_opt(model, test_poison_loader, criterion, GFT_noise):
    device = torch.device("cuda" if args.cuda else "cpu")

    test_loss = 0.0
    count = 0.0
    model.eval()
    test_pred = []
    test_true = []
    for data, label, v, _ in test_poison_loader:
        data, label = data.to(device), label.to(device).squeeze()
        v = v.to(device)
        GFT_noise = GFT_noise.to(device)
        data = data.permute(0, 2, 1)
        label.fill_(target_label)
        for kkkk in range(len(data)):
            data[kkkk] = GFT_opt(data[kkkk].unsqueeze(0).permute(0, 2, 1), GFT_noise,v[kkkk].unsqueeze(0)).permute(0, 2, 1)
            label[kkkk] = torch.tensor([target_label])
        batch_size = data.size()[0]
        logits = model(data.float())
        label = label.to(torch.int64)
        loss = criterion(logits, label)
        preds = logits.max(dim=1)[1]
        count += batch_size
        test_loss += loss.item() * batch_size
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    poison_test_acc = metrics.accuracy_score(test_true, test_pred)
    poison_avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    print('poison_test_acc:{}'.format(poison_test_acc))
    print('poison_avg_per_class_acc:{}'.format(poison_avg_per_class_acc))
    return poison_test_acc


def test_poison_opt2(args, io):
    test_poison_loader = DataLoader(
        ModelNet40(partition='test', num_points=args.num_points, test_poison_target=0, target_num=None,
                   model=None, test_poison=True, GFT_noise=None), num_workers=0,
        batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    GFT_noise = np.load('pretrained-results\\DGCNN\\GFT_noise epoc249,plb0,phb0,pls1024,phs1024,a_lr0.01, inl0.1, fpTrue, pr0.05, wb0.5,wa0.5,pwa0.9,pwd0.1.npy')
    GFT_noise = torch.tensor(GFT_noise)
    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = DGCNN(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_loss = 0.0
    count = 0.0
    model.eval()
    test_pred = []
    test_true = []
    for data, label, v, _ in test_poison_loader:
        data, label = data.to(device), label.to(device).squeeze()
        v = v.to(device)
        GFT_noise = GFT_noise.to(device)
        data = data.permute(0, 2, 1)
        label.fill_(target_label)

        data_clean = data[0].clone()
        ###############
        for kkkk in range(len(data)):
            data[kkkk] = GFT_opt(data[kkkk].unsqueeze(0).permute(0, 2, 1), GFT_noise,v[kkkk].unsqueeze(0)).permute(0, 2, 1)
            label[kkkk] = torch.tensor([target_label])
        ### visualization
        #show_pl(data[0].permute(1, 0).detach().cpu().numpy(), [None])
        print('l2 distance of one point cloud:{}'.format(torch.norm(data[0] - data_clean)))

        batch_size = data.size()[0]
        logits = model(data.float())
        label = label.to(torch.int64)
        preds = logits.max(dim=1)[1]
        count += batch_size
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    poison_test_acc = metrics.accuracy_score(test_true, test_pred)
    print('poison_test_acc:{}'.format(poison_test_acc))
    return poison_test_acc

def test_poison(model, test_poison_loader, criterion):
    device = torch.device("cuda" if args.cuda else "cpu")

    test_loss = 0.0
    count = 0.0
    model.eval()
    test_pred = []
    test_true = []
    for data, label in test_poison_loader:
        data, label = data.to(device), label.to(device).squeeze()


        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        label = label.to(torch.int64)
        loss = criterion(logits, label)
        preds = logits.max(dim=1)[1]
        count += batch_size
        test_loss += loss.item() * batch_size
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    poison_test_acc = metrics.accuracy_score(test_true, test_pred)
    poison_avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    print('poison_test_acc:{}'.format(poison_test_acc))
    print('poison_avg_per_class_acc:{}'.format(poison_avg_per_class_acc))




def test(args, io):
    test_poison_loader = DataLoader(
        ModelNet40(partition='test', num_points=args.num_points, test_poison_target=0, target_num=None,
                   model=None, test_poison=True, GFT_noise=None), num_workers=0,
        batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = DGCNN(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_poison_loader:
        #show_pl(data[0].detach().cpu().numpy(), [None])

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',  #250
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=True,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='pretrained-results\\DGCNN\\epoc249 model.t7', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test_poison_opt2(args, io)
