# -*- coding: utf-8 -*-
import torch
import torchvision
from torch.autograd import Variable
from utils import imshow, calc_loss
import matplotlib.pyplot as plt
import numpy as np
from dataloaders import args, train_loader, test_loader
import torch.nn.functional as F


def pre_train_test(model, optimizer, criterion):
    alpha_num = args.uniform_approx_num
    h_u_sum = 0
    exp_h_u_sum = 0
    exp_alpha_sums = [0 for j in range(alpha_num + 1)]
    exp_alpha_prod_sums = [0 for j in range(alpha_num + 1)]
    grad_sum = 0
    exp_grad_sum = 0
    count = 0

    for i in range(args.approx_num):
        h_u, grad_square = pre_train_test_iteration(model, optimizer, criterion, i)
        exp_h_u_val = np.exp(h_u) - 1

        for j in range(alpha_num + 1):
            alpha = j / alpha_num
            exp_alpha_val = np.exp(alpha * h_u)
            exp_alpha_sums[j] += exp_alpha_val
            exp_alpha_prod_sums[j] += exp_alpha_val * grad_square * h_u

        h_u_sum += h_u
        exp_h_u_sum += exp_h_u_val
        grad_sum += grad_square
        exp_grad_sum += exp_h_u_val * grad_square
        count += 1

    h_u_mean = h_u_sum / count
    h_u_exp_mean = exp_h_u_sum / count
    grad_mean = grad_sum / count
    exp_grad_mean = exp_grad_sum / count
    h_u_plus_exp_grad_mean = h_u_mean + 2 * exp_grad_mean

    h_u_result = h_u_mean / len(train_loader)
    h_u_exp_result = h_u_exp_mean / len(train_loader)
    grad_result = grad_mean / len(train_loader)
    exp_grad_result = exp_grad_mean / len(train_loader)
    h_u_plus_exp_grad_result = h_u_plus_exp_grad_mean / len(train_loader)

    results = (h_u_result, h_u_exp_result, grad_result, exp_grad_result, h_u_plus_exp_grad_result)
    # results = (h_u_result.item(), h_u_exp_result.item(), grad_result.item(), exp_grad_result.item(), h_u_plus_exp_grad_result.item())

    exp_alpha_means = [val/count for val in exp_alpha_sums]
    exp_alpha_mean = np.sum(exp_alpha_means)/len(exp_alpha_means)
    exp_alpha_prod_means = [val/count for val in exp_alpha_prod_sums]
    exp_alpha_prod_mean = np.sum(exp_alpha_prod_means)/len(exp_alpha_prod_means)
    exp_alpha_frac_means = [val/exp_alpha_means[j] for j, val in enumerate(exp_alpha_prod_means)]
    exp_alpha_frac_mean = np.sum(exp_alpha_frac_means)/len(exp_alpha_frac_means)

    exp_alpha_result = exp_alpha_mean / len(train_loader)
    exp_alpha_prod_result = exp_alpha_prod_mean / len(train_loader)
    exp_alpha_frac_result = exp_alpha_frac_mean / len(train_loader)

    results_alpha = (exp_alpha_result, exp_alpha_prod_result, exp_alpha_frac_result)

    return results, results_alpha


def pre_train_test_iteration(model, optimizer, criterion, iteration):

    model.auto_update_noise(False)
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


def post_train_test(model, optimizer, criterion):

    count = 0
    loss_sum = 0
    norm_sum = 0

    for i in range(args.approx_num):
        loss, norm = post_train_test_iteration(model, optimizer, criterion, i)

        count += 1
        loss_sum += loss
        norm_sum += norm

    loss_mean = loss_sum / count
    norm_mean = norm_sum / count

    results = (loss_mean, norm_mean)
    # results = (loss_mean, norm_mean.item())

    return results


def post_train_test_iteration(model, optimizer, criterion, iteration):

    model.auto_update_noise(False)
    model.train()
    count = 0
    loss_sum = 0
    norm_sum = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        loss = calc_loss(model, optimizer, criterion, data, target)
        loss_data = loss.data[0]

        # update sums
        count += 1
        loss_sum += loss_data
        norm_val = 0
        params = list(model.named_parameters())
        for i, (name, p) in enumerate(params):
            if str(name).count("bias"):
                continue
            norm_val += torch.sum(p.data ** 4)

        norm_sum += norm_val


        if batch_idx % args.log_interval == 0:
            print('pre train test iteration: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                iteration, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_data))

    loss_mean = loss_sum / count
    norm_mean = norm_sum / count

    return loss_mean, norm_mean


def test(model, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
