from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloaders import args
from models import GaussDropoutNet, GaussDropoutNetLimited\
    , BernoulliDropoutNet, BernoulliDropoutNetLimited
from test import pre_train_test
from datetime import datetime
from utils import save_fig, count_parameters
import numpy as np


def run_aux(model_class, additional_fc_layers=2):

    model = model_class(additional_fc_layers=additional_fc_layers)

    num_layers = 4 + additional_fc_layers
    parameters_num = count_parameters(model)

    if args.cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = F.nll_loss

    h_u_mean, h_u_exp_mean, grad_mean, exp_grad_mean, total_mean = pre_train_test(model, optimizer, criterion)
    print(str(num_layers) + ', ' + str(parameters_num) + ', ' + str(h_u_mean) + ', ' + str(h_u_exp_mean) + ', ' + str(grad_mean)
          + ', ' + str(exp_grad_mean) + ', ' + str(total_mean))

    return parameters_num, h_u_mean, h_u_exp_mean, grad_mean, exp_grad_mean, total_mean


def run_gauss_dropout(min_fc_layers=0, max_fc_layers=args.max_fc_layers):

    print("num_layers, parameters_num, h_u_mean , h_u_exp_mean , grad_mean , exp_grad_mean , total_mean")
    num_layers_list = []
    parameters_num_list = []
    h_u_mean_list = []
    h_u_exp_mean_list = []
    grad_mean_list = []

    exp_grad_mean_list = []
    total_mean_list = []

    for i in range(min_fc_layers, max_fc_layers):
        parameters_num, h_u_mean, h_u_exp_mean, grad_mean, exp_grad_mean, total_mean\
            = run_aux(GaussDropoutNet, additional_fc_layers=i)
        num_layers = i + 4

        num_layers_list.append(num_layers)
        parameters_num_list.append(parameters_num)
        h_u_mean_list.append(h_u_mean)
        h_u_exp_mean_list.append(h_u_exp_mean)
        grad_mean_list.append(grad_mean)
        exp_grad_mean_list.append(exp_grad_mean)
        total_mean_list.append(total_mean)

    print("num_layers_gauss")
    print(num_layers_list)
    print("parameters_num_gauss")
    print(parameters_num_list)
    print("h_u_mean_gauss")
    print(h_u_mean_list)
    print("h_u_exp_mean_gauss")
    print(h_u_exp_mean_list)
    print("grad_mean_gauss")
    print(grad_mean_list)
    print("exp_grad_mean_gauss")
    print(exp_grad_mean_list)
    print("total_mean_gauss")
    print(total_mean_list)

    y_list = [h_u_mean_list, h_u_exp_mean_list, grad_mean_list, exp_grad_mean_list, total_mean_list]
    series_labels = ["f1=h_u", "f2=exp(h_u)-1", "f3=||grad_u(l)||^2", "f2*f3", "f1+2*f2*f3"]
    title = "Gaussian Dropout, 2500 parameters per layer"
    ylabel = "ampirical average"
    xlabel = "number of layers"

    save_fig(x=num_layers_list, y_list=y_list, series_labels=series_labels, title=title, xlabel=xlabel, ylabel=ylabel)

    run_bernoulli_dropout(gauss_results=y_list)


def run_gauss_dropout_limited(min_fc_layers=0, max_fc_layers=args.max_fc_layers):

    print("num_layers, parameters_num, h_u_mean , h_u_exp_mean , grad_mean , exp_grad_mean , total_mean")
    num_layers_list = []
    parameters_num_list = []
    h_u_mean_list = []
    h_u_exp_mean_list = []
    grad_mean_list = []
    exp_grad_mean_list = []
    total_mean_list = []

    for i in range(min_fc_layers, max_fc_layers):
        parameters_num, h_u_mean, h_u_exp_mean, grad_mean, exp_grad_mean, total_mean\
            = run_aux(GaussDropoutNetLimited, additional_fc_layers=i)
        num_layers = i + 4

        num_layers_list.append(num_layers)
        parameters_num_list.append(parameters_num)
        h_u_mean_list.append(h_u_mean)
        h_u_exp_mean_list.append(h_u_exp_mean)
        grad_mean_list.append(grad_mean)
        exp_grad_mean_list.append(exp_grad_mean)
        total_mean_list.append(total_mean)

    print("num_layers_gauss")
    print(num_layers_list)
    print("parameters_num_gauss")
    print(parameters_num_list)
    print("h_u_mean_gauss")
    print(h_u_mean_list)
    print("h_u_exp_mean_gauss")
    print(h_u_exp_mean_list)
    print("grad_mean_gauss")
    print(grad_mean_list)
    print("exp_grad_mean_gauss")
    print(exp_grad_mean_list)
    print("total_mean_gauss")
    print(total_mean_list)

    paramrters_mean = np.sum(parameters_num_list)/len(parameters_num_list)
    y_list = [h_u_mean_list, h_u_exp_mean_list, grad_mean_list, exp_grad_mean_list, total_mean_list]
    series_labels = ["f1=h_u", "f2=exp(h_u)-1", "f3=||grad_u(l)||^2", "f2*f3", "f1+2*f2*f3"]
    title = "Gaussian Dropout,  ~" + str(paramrters_mean) + " parameters total"
    ylabel = "ampirical average"
    xlabel = "number of layers"

    save_fig(x=num_layers_list, y_list=y_list, series_labels=series_labels, title=title, xlabel=xlabel, ylabel=ylabel)

    run_bernoulli_dropout_limited(gauss_results=y_list)


def run_bernoulli_dropout(gauss_results, min_fc_layers=0, max_fc_layers=args.max_fc_layers):

    print("num_layers, parameters_num, h_u_mean , h_u_exp_mean , grad_mean , exp_grad_mean , total_mean")
    num_layers_list = []
    parameters_num_list = []
    h_u_mean_list = []
    h_u_exp_mean_list = []
    grad_mean_list = []
    exp_grad_mean_list = []
    total_mean_list = []

    for i in range(min_fc_layers, max_fc_layers):
        parameters_num, h_u_mean, h_u_exp_mean, grad_mean, exp_grad_mean, total_mean\
            = run_aux(BernoulliDropoutNet, additional_fc_layers=i)
        num_layers = i + 4

        num_layers_list.append(num_layers)
        parameters_num_list.append(parameters_num)
        h_u_mean_list.append(h_u_mean)
        h_u_exp_mean_list.append(h_u_exp_mean)
        grad_mean_list.append(grad_mean)
        exp_grad_mean_list.append(exp_grad_mean)
        total_mean_list.append(total_mean)

    print("num_layers_bernoulli")
    print(num_layers_list)
    print("parameters_num_bernoulli")
    print(parameters_num_list)
    print("h_u_mean_bernoulli")
    print(h_u_mean_list)
    print("h_u_exp_mean_bernoulli")
    print(h_u_exp_mean_list)
    print("grad_mean_bernoulli")
    print(grad_mean_list)
    print("exp_grad_mean_bernoulli")
    print(exp_grad_mean_list)
    print("total_mean_bernoulli")
    print(total_mean_list)

    bernoulli_results = [h_u_mean_list, h_u_exp_mean_list, grad_mean_list, exp_grad_mean_list, total_mean_list]
    compare_results = []
    for i, gauss_list in enumerate(gauss_results):
        bernoulli_list = bernoulli_results[i]
        compare_list = []

        for j, gauss_val in enumerate(gauss_list):
            bernoulli_val = bernoulli_list[j]
            compare_list.append(np.abs(gauss_val - bernoulli_val))

        compare_results.append(compare_list)

    print("h_u_mean_diff")
    print(compare_results[0])
    print("h_u_exp_mean_diff")
    print(compare_results[1])
    print("grad_mean_diff")
    print(compare_results[2])
    print("exp_grad_mean_diff")
    print(compare_results[3])
    print("total_mean_diff")
    print(compare_results[4])

    y_list = compare_results
    series_labels = ["f1=h_u", "f2=exp(h_u)-1", "f3=||grad_u(l)||^2", "f2*f3", "f1+2*f2*f3"]
    title = "comparing Gaussian and Bernoulli Dropout, 2500 parameters per layer"
    ylabel = "ampirical averages difference in absolute value"
    xlabel = "number of layers"

    save_fig(x=num_layers_list, y_list=y_list, series_labels=series_labels, title=title, xlabel=xlabel, ylabel=ylabel)


def run_bernoulli_dropout_limited(gauss_results, min_fc_layers=0, max_fc_layers=args.max_fc_layers):

    print("num_layers, parameters_num, h_u_mean , h_u_exp_mean , grad_mean , exp_grad_mean , total_mean")
    num_layers_list = []
    parameters_num_list = []
    h_u_mean_list = []
    h_u_exp_mean_list = []
    grad_mean_list = []
    exp_grad_mean_list = []
    total_mean_list = []

    for i in range(min_fc_layers, max_fc_layers):
        parameters_num, h_u_mean, h_u_exp_mean, grad_mean, exp_grad_mean, total_mean\
            = run_aux(BernoulliDropoutNetLimited, additional_fc_layers=i)
        num_layers = i + 4

        num_layers_list.append(num_layers)
        parameters_num_list.append(parameters_num)
        h_u_mean_list.append(h_u_mean)
        h_u_exp_mean_list.append(h_u_exp_mean)
        grad_mean_list.append(grad_mean)
        exp_grad_mean_list.append(exp_grad_mean)
        total_mean_list.append(total_mean)

    print("num_layers_bernoulli")
    print(num_layers_list)
    print("parameters_num_bernoulli")
    print(parameters_num_list)
    print("h_u_mean_bernoulli")
    print(h_u_mean_list)
    print("h_u_exp_mean_bernoulli")
    print(h_u_exp_mean_list)
    print("grad_mean_bernoulli")
    print(grad_mean_list)
    print("exp_grad_mean_bernoulli")
    print(exp_grad_mean_list)
    print("total_mean_bernoulli")
    print(total_mean_list)

    paramrters_mean = np.sum(parameters_num_list)/len(parameters_num_list)

    bernoulli_results = [h_u_mean_list, h_u_exp_mean_list, grad_mean_list, exp_grad_mean_list, total_mean_list]
    compare_results = []
    for i, gauss_list in enumerate(gauss_results):
        bernoulli_list = bernoulli_results[i]
        compare_list = []

        for j, gauss_val in enumerate(gauss_list):
            bernoulli_val = bernoulli_list[j]
            compare_list.append(np.abs(gauss_val - bernoulli_val))

        compare_results.append(compare_list)

    print("h_u_mean_diff")
    print(compare_results[0])
    print("h_u_exp_mean_diff")
    print(compare_results[1])
    print("grad_mean_diff")
    print(compare_results[2])
    print("exp_grad_mean_diff")
    print(compare_results[3])
    print("total_mean_diff")
    print(compare_results[4])

    y_list = compare_results
    series_labels = ["f1=h_u", "f2=exp(h_u)-1", "f3=||grad_u(l)||^2", "f2*f3", "f1+2*f2*f3"]
    title = "comparing Gaussian and Bernoulli Dropout, ~" + str(paramrters_mean) + " parameters total"
    ylabel = "ampirical averages difference in absolute value"
    xlabel = "number of layers"

    save_fig(x=num_layers_list, y_list=y_list, series_labels=series_labels, title=title, xlabel=xlabel, ylabel=ylabel)


########################################################################
start_time = datetime.now()

print("args")
print(args)

if args.limited_params:
    run_gauss_dropout_limited()
else:
    run_gauss_dropout()

end_time = datetime.now()
print("run time: " + str(end_time - start_time))
