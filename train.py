# -*- coding: utf-8 -*-
import torch
import torchvision
from torch.autograd import Variable
from utils import imshow, calc_loss
import matplotlib.pyplot as plt
import numpy as np
from dataloaders import args, train_loader
from test import test
import torch.nn.functional as F


def train(model, optimizer, criterion):
    for epoch in range(1, args.epochs + 1):
        print("train_iteration")
        train_iteration(epoch, model, optimizer, criterion)
        print("test")
        test(model, criterion)
    print("finished training")


def train_iteration(epoch, model, optimizer, criterion):
    model.auto_update_noise(True)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        loss = calc_loss(model, optimizer, criterion, data, target)

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
