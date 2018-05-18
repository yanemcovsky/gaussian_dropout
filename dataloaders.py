# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--approx-num', type=int, default=10, metavar='N',
                    help='number of data points used for approximating a mean')
parser.add_argument('--max-fc-layers', type=int, default=10, metavar='N',
                    help='maximal number of additional fc layers to add to networks, '
                         'not including first and lest fc layers')
parser.add_argument('--limit-params', action='store_true', default=False,
                    help='limit number of parameters when increasing number of layers')
parser.add_argument('--uniform-approx-num', type=int, default=1000, metavar='N',
                    help='number of data points used for approximating a mean for a uniform distribution')
parser.add_argument('--mnist-databse', action='store_true', default=False,
                    help='use mnist database, otherwise will use CIFAR10')
parser.add_argument('--test-before-train', action='store_true', default=True,
                    help='run the pre-train test')
parser.add_argument('--test-after-train', action='store_true', default=False,
                    help='train and run the post-train test')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if args.mnist_databse:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

trainset = None
if args.mnist_databse:
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
else:
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, **kwargs)

testset = None
if args.mnist_databse:
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
else:
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                         shuffle=True, **kwargs)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
