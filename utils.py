# -*- coding: utf-8 -*-
import torch
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import numpy as np
from torch.autograd import Variable
from dataloaders import args

def gauss_var(p):
    return (1-p)/p


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def fix_params_layer_size(l):
    if args.mnist_databse:
        if l == 0:
            return 65
        sqrt_arg = (860 * l + 1089)
        sqrt = np.sqrt(sqrt_arg)
        total = 5 * (sqrt - 33) / l
        return int(total)

    if l == 0:
        return int(138.28)
    sqrt_arg = (17976 * l + 4225)
    sqrt = np.sqrt(sqrt_arg)
    total = (sqrt - 65) / l
    return int(total)


def save_fig(x, y_list, series_labels, title, xlabel, ylabel):

    # markers_unique = ["x", "o", "v", "d", ",", "^", "*", "|"]
    markers_unique = ["x", "o", "d", ",", "^"]
    markers_repeats = int(np.floor(len(y_list)/len(markers_unique))) + 1
    markers = markers_unique * markers_repeats

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for i, y in enumerate(y_list):
        label = series_labels[i]
        ax1.scatter(x, y, s=10, label=label, marker=markers[i])
        ax1.plot(x, y)
    plt.legend(loc='upper right')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale('log')
    plt.title(title)
    plt.savefig("saved_figures/" + title + ".png")
    plt.show()


def count_parameters(model):
    # return sum(p.numel() for p in model.parameters() if p.requires_grad and not str(name).count("bias"))
    count = 0
    params = list(model.named_parameters())
    for (name, p) in params:
        if p.requires_grad and not str(name).count("bias"):
            count += p.numel()
    return count




def calc_loss(model, optimizer, criterion, data, target):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()

    return loss
