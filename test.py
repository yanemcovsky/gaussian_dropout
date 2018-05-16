# -*- coding: utf-8 -*-
import torch
import torchvision
from torch.autograd import Variable
from utils import imshow
import matplotlib.pyplot as plt
import numpy as np
from dataloaders import args, train_loader
import torch.nn.functional as F


def pre_train_test(model, optimizer, criterion):

    h_u_sum = 0
    h_u_exp_sum = 0
    grad_sum = 0
    exp_grad_sum = 0
    count = 0

    for i in range(args.approx_num):
        h_u, grad_square = pre_train_test_iteration(model, optimizer, criterion, i)
        h_u_exp_val = np.exp(h_u) - 1

        h_u_sum += h_u
        h_u_exp_sum += h_u_exp_val
        grad_sum += grad_square
        exp_grad_sum += h_u_exp_val * grad_square
        count += 1

    h_u_mean = h_u_sum / count
    h_u_exp_mean = h_u_exp_sum / count
    grad_mean = grad_sum / count
    exp_grad_mean = exp_grad_sum / count

    total_mean = h_u_mean + 2 * exp_grad_mean
    return h_u_mean, h_u_exp_mean, grad_mean, exp_grad_mean, total_mean


def pre_train_test_iteration(model, optimizer, criterion, iteration):

    model.train()
    max_loss = -1
    max_arg = None

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        loss = calc_loss(model, optimizer, criterion, data, target)
        loss_data = loss.data[0]

        # update max_loss
        if loss_data > max_loss:
            max_loss = loss_data
            max_arg = (data, target)

        if batch_idx % args.log_interval == 0:
            print('pre train test iteration: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                iteration, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_data))

    data, target = max_arg
    loss = calc_loss(model, optimizer, criterion, data, target)
    grad_square = 0
    params = list(model.named_parameters())
    for i, (name, p) in enumerate(params):
        if str(name).count("bias"):
            continue
        grad_square += torch.sum(p.grad.data ** 2)

    h_u = max_loss ** 2
    return h_u, grad_square


def calc_loss(model, optimizer, criterion, data, target):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()

    return loss
