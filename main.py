from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloaders import args
from models import GaussDropoutNet, GaussDropoutNetLimited\
    , BernoulliDropoutNet, BernoulliDropoutNetLimited
from test import pre_train_test, post_train_test
from datetime import datetime
from utils import save_fig, count_parameters
import numpy as np
from train import train


def run_model(model_class, additional_fc_layers=2, test_before_train=True, test_after_train=True):

    model = model_class(additional_fc_layers=additional_fc_layers)

    num_layers = 4 + additional_fc_layers
    parameters_num = count_parameters(model)

    if args.cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = F.nll_loss

    pre_train_results = None
    if test_before_train:
        print("pre_train_test")
        results, results_alpha = pre_train_test(model, optimizer, criterion)
        h_u_mean, h_u_exp_mean, grad_mean, exp_grad_mean, h_u_plus_exp_grad_mean = results
        exp_alpha_mean, exp_alpha_prod_mean, exp_alpha_frac_mean = results_alpha

        print(str(num_layers) + ', ' + str(parameters_num) + ', ' + str(h_u_mean) + ', ' + str(h_u_exp_mean)
              + ', ' + str(grad_mean)+ ', ' + str(exp_grad_mean) + ', ' + str(h_u_plus_exp_grad_mean)
              + ', ' + str(exp_alpha_mean) + ', ' + str(exp_alpha_prod_mean) + ', ' + str(exp_alpha_frac_mean))

        pre_train_results = h_u_mean, h_u_exp_mean, grad_mean, exp_grad_mean, h_u_plus_exp_grad_mean\
            , exp_alpha_mean, exp_alpha_prod_mean, exp_alpha_frac_mean

    post_train_results = None
    if test_after_train:
        train(model, optimizer, criterion)
        print("post_train_test")
        post_train_results = post_train_test(model, optimizer, criterion)

    return num_layers, parameters_num, pre_train_results, post_train_results


def run_gauss(min_fc_layers=0, max_fc_layers=args.max_fc_layers
              , limit_params=False, test_before_train=True, test_after_train=True):

    print("run_gauss with limit_params=" + str(limit_params))

    model = GaussDropoutNet
    if limit_params:
        model = GaussDropoutNetLimited

    print("num_layers, parameters_num, h_u_mean , h_u_exp_mean , grad_mean , exp_grad_mean , h_u_plus_exp_grad_mean"
          " , exp_alpha_mean , exp_alpha_prod_mean , exp_alpha_frac_mean")
    num_layers_list = []
    parameters_num_list = []
    h_u_mean_list = []
    h_u_exp_mean_list = []
    grad_mean_list = []

    exp_grad_mean_list = []
    h_u_plus_exp_grad_mean_list = []

    exp_alpha_mean_list = []
    exp_alpha_prod_mean_list = []
    exp_alpha_frac_mean_list = []

    loss_mean_list = []
    norm_mean_list = []

    for i in range(min_fc_layers, max_fc_layers):
        num_layers, parameters_num, pre_train_results, post_train_results = run_model(model,
                                                                                      additional_fc_layers=i,
                                                                                      test_before_train=test_before_train,
                                                                                      test_after_train=test_after_train)

        num_layers_list.append(num_layers)
        parameters_num_list.append(parameters_num)

        if test_before_train:
            h_u_mean, h_u_exp_mean, grad_mean, exp_grad_mean, h_u_plus_exp_grad_mean\
                , exp_alpha_mean, exp_alpha_prod_mean, exp_alpha_frac_mean = pre_train_results

            h_u_mean_list.append(h_u_mean)
            h_u_exp_mean_list.append(h_u_exp_mean)
            grad_mean_list.append(grad_mean)
            exp_grad_mean_list.append(exp_grad_mean)
            h_u_plus_exp_grad_mean_list.append(h_u_plus_exp_grad_mean)

            exp_alpha_mean_list.append(exp_alpha_mean)
            exp_alpha_prod_mean_list.append(exp_alpha_prod_mean)
            exp_alpha_frac_mean_list.append(exp_alpha_frac_mean)

        if test_after_train:
            loss_mean, norm_mean = post_train_results

            loss_mean_list.append(loss_mean)
            norm_mean_list.append(norm_mean)

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
    print("h_u_plus_exp_grad_mean_gauss")
    print(h_u_plus_exp_grad_mean_list)

    print("exp_alpha_mean_gauss")
    print(exp_alpha_mean_list)
    print("exp_alpha_prod_mean_gauss")
    print(exp_alpha_prod_mean_list)
    print("exp_alpha_frac_mean_gauss")
    print(exp_alpha_frac_mean_list)

    print("loss_mean_list")
    print(loss_mean_list)
    print("norm_mean_list")
    print(norm_mean_list)

    paramrters_mean = int(np.sum(parameters_num_list)/len(parameters_num_list))

    if test_before_train:
        y_list = [h_u_mean_list, h_u_exp_mean_list, grad_mean_list, exp_alpha_mean_list,
                  exp_grad_mean_list, h_u_plus_exp_grad_mean_list,
                  exp_alpha_prod_mean_list, exp_alpha_frac_mean_list]
        series_labels = ["f1=h_u", "f2=exp(h_u)-1", "f3=||grad_u(l)||^2", "f4=exp(a*h_u)",
                         "E[f2*f3]", "E[f1+2*f2*f3]", "E[f1*f3*f4]", "E[f1*f3*f4]/E[f4]"]
        ylabel = "c"
        xlabel = "number of layers"

        title = "Gaussian Dropout, "
        if limit_params:
            title += "~" + str(paramrters_mean) + " parameters total"
        else:
            title += "2500 parameters per layer"

        save_fig(x=num_layers_list, y_list=y_list, series_labels=series_labels, title=title, xlabel=xlabel, ylabel=ylabel)

    if test_after_train:
        gauss_compare_results = [loss_mean_list, norm_mean_list]
        run_bernoulli(gauss_results=gauss_compare_results, limit_params=limit_params)


def run_gauss_dropout(min_fc_layers=0, max_fc_layers=args.max_fc_layers
                      , test_before_train=True, test_after_train=True):
    run_gauss(min_fc_layers=min_fc_layers, max_fc_layers=max_fc_layers
              , limit_params=False, test_before_train=test_before_train, test_after_train=test_after_train)

    # print("num_layers, parameters_num, h_u_mean , h_u_exp_mean , grad_mean , exp_grad_mean , h_u_plus_exp_grad_mean"
    #       " , exp_alpha_mean , exp_alpha_prod_mean , exp_alpha_frac_mean")
    # num_layers_list = []
    # parameters_num_list = []
    # h_u_mean_list = []
    # h_u_exp_mean_list = []
    # grad_mean_list = []
    #
    # exp_grad_mean_list = []
    # h_u_plus_exp_grad_mean_list = []
    #
    # for i in range(min_fc_layers, max_fc_layers):
    # pre_train_results, post_train_results = run_model(model, additional_fc_layers=i)
    #
    # num_layers, parameters_num \
    #     , h_u_mean, h_u_exp_mean, grad_mean, exp_grad_mean, h_u_plus_exp_grad_mean \
    #     , exp_alpha_mean, exp_alpha_prod_mean, exp_alpha_frac_mean \
    #     = pre_train_results
    #
    #     num_layers_list.append(num_layers)
    #     parameters_num_list.append(parameters_num)
    #     h_u_mean_list.append(h_u_mean)
    #     h_u_exp_mean_list.append(h_u_exp_mean)
    #     grad_mean_list.append(grad_mean)
    #     exp_grad_mean_list.append(exp_grad_mean)
    #     h_u_plus_exp_grad_mean_list.append(h_u_plus_exp_grad_mean)
    #
    # print("num_layers_gauss")
    # print(num_layers_list)
    # print("parameters_num_gauss")
    # print(parameters_num_list)
    # print("h_u_mean_gauss")
    # print(h_u_mean_list)
    # print("h_u_exp_mean_gauss")
    # print(h_u_exp_mean_list)
    # print("grad_mean_gauss")
    # print(grad_mean_list)
    # print("exp_grad_mean_gauss")
    # print(exp_grad_mean_list)
    # print("h_u_plus_exp_grad_mean_gauss")
    # print(h_u_plus_exp_grad_mean_list)
    #
    # y_list = [h_u_mean_list, h_u_exp_mean_list, grad_mean_list, exp_grad_mean_list, h_u_plus_exp_grad_mean_list]
    # series_labels = ["f1=h_u", "f2=exp(h_u)-1", "f3=||grad_u(l)||^2", "f2*f3", "f1+2*f2*f3"]
    # title = "Gaussian Dropout, 2500 parameters per layer"
    # ylabel = "ampirical average"
    # xlabel = "number of layers"
    #
    # save_fig(x=num_layers_list, y_list=y_list, series_labels=series_labels, title=title, xlabel=xlabel, ylabel=ylabel)
    #
    # run_bernoulli_dropout(gauss_results=y_list)


def run_gauss_dropout_limited(min_fc_layers=0, max_fc_layers=args.max_fc_layers
                              , test_before_train=True, test_after_train=True):

    run_gauss(min_fc_layers=min_fc_layers, max_fc_layers=max_fc_layers
              , limit_params=True, test_before_train=test_before_train, test_after_train=test_after_train)

    # print("num_layers, parameters_num, h_u_mean , h_u_exp_mean , grad_mean , exp_grad_mean , h_u_plus_exp_grad_mean")
    # num_layers_list = []
    # parameters_num_list = []
    # h_u_mean_list = []
    # h_u_exp_mean_list = []
    # grad_mean_list = []
    # exp_grad_mean_list = []
    # h_u_plus_exp_grad_mean_list = []
    #
    # for i in range(min_fc_layers, max_fc_layers):
    # pre_train_results, post_train_results = run_model(model, additional_fc_layers=i)
    #
    # num_layers, parameters_num \
    #     , h_u_mean, h_u_exp_mean, grad_mean, exp_grad_mean, h_u_plus_exp_grad_mean \
    #     , exp_alpha_mean, exp_alpha_prod_mean, exp_alpha_frac_mean \
    #     = pre_train_results
    #
    #     num_layers_list.append(num_layers)
    #     parameters_num_list.append(parameters_num)
    #     h_u_mean_list.append(h_u_mean)
    #     h_u_exp_mean_list.append(h_u_exp_mean)
    #     grad_mean_list.append(grad_mean)
    #     exp_grad_mean_list.append(exp_grad_mean)
    #     h_u_plus_exp_grad_mean_list.append(h_u_plus_exp_grad_mean)
    #
    # print("num_layers_gauss")
    # print(num_layers_list)
    # print("parameters_num_gauss")
    # print(parameters_num_list)
    # print("h_u_mean_gauss")
    # print(h_u_mean_list)
    # print("h_u_exp_mean_gauss")
    # print(h_u_exp_mean_list)
    # print("grad_mean_gauss")
    # print(grad_mean_list)
    # print("exp_grad_mean_gauss")
    # print(exp_grad_mean_list)
    # print("h_u_plus_exp_grad_mean_gauss")
    # print(h_u_plus_exp_grad_mean_list)
    #
    # paramrters_mean = np.sum(parameters_num_list)/len(parameters_num_list)
    # y_list = [h_u_mean_list, h_u_exp_mean_list, grad_mean_list, exp_grad_mean_list, h_u_plus_exp_grad_mean_list]
    # series_labels = ["f1=h_u", "f2=exp(h_u)-1", "f3=||grad_u(l)||^2", "f2*f3", "f1+2*f2*f3"]
    # title = "Gaussian Dropout,  ~" + str(paramrters_mean) + " parameters total"
    # ylabel = "ampirical average"
    # xlabel = "number of layers"
    #
    # save_fig(x=num_layers_list, y_list=y_list, series_labels=series_labels, title=title, xlabel=xlabel, ylabel=ylabel)
    #
    # run_bernoulli_dropout_limited(gauss_results=y_list)


def run_bernoulli(gauss_results, min_fc_layers=0, max_fc_layers=args.max_fc_layers
                  , limit_params=False):

    print("run_bernoulli with limit_params=" + str(limit_params))

    model = BernoulliDropoutNet
    if limit_params:
        model = BernoulliDropoutNetLimited

    num_layers_list = []
    parameters_num_list = []

    loss_mean_list = []
    norm_mean_list = []

    for i in range(min_fc_layers, max_fc_layers):
        num_layers, parameters_num, pre_train_results, post_train_results = run_model(model,
                                                                                      additional_fc_layers=i,
                                                                                      test_before_train=False,
                                                                                      test_after_train=True)

        loss_mean, norm_mean = post_train_results

        num_layers_list.append(num_layers)
        parameters_num_list.append(parameters_num)

        loss_mean_list.append(loss_mean)
        norm_mean_list.append(norm_mean)

    print("num_layers_gauss")
    print(num_layers_list)
    print("parameters_num_gauss")
    print(parameters_num_list)
    print("loss_mean_list")
    print(loss_mean_list)
    print("norm_mean_list")
    print(norm_mean_list)

    bernoulli_results = [loss_mean_list, norm_mean_list]

    compare_results = []
    for i, gauss_list in enumerate(gauss_results):
        bernoulli_list = bernoulli_results[i]
        compare_list = []

        for j, gauss_val in enumerate(gauss_list):
            bernoulli_val = bernoulli_list[j]
            compare_list.append(np.abs(gauss_val - bernoulli_val))

        compare_results.append(compare_list)

    print("loss_mean_diff")
    print(compare_results[0])
    print("norm_mean_diff")
    print(compare_results[1])

    paramrters_mean = int(np.sum(parameters_num_list)/len(parameters_num_list))
    y_list = compare_results
    series_labels = ["f1=h_u", "f2=exp(h_u)-1", "f3=||grad_u(l)||^2", "f4=exp(a*h_u)",
                     "E[f2*f3]", "E[f1+2*f2*f3]", "E[f1*f3*f4]", "E[f1*f3*f4]/E[f4]"]
    ylabel = "ampirical averages difference in absolute value"
    xlabel = "number of layers"

    title = "comparing Gaussian and Bernoulli Dropout, "
    if limit_params:
        title += "~" + str(paramrters_mean) + " parameters total"
    else:
        title += "2500 parameters per layer"

    save_fig(x=num_layers_list, y_list=y_list, series_labels=series_labels, title=title, xlabel=xlabel, ylabel=ylabel)


def run_bernoulli_dropout(gauss_results, min_fc_layers=0, max_fc_layers=args.max_fc_layers):
    run_bernoulli(gauss_results=gauss_results, min_fc_layers=min_fc_layers, max_fc_layers=max_fc_layers
                  , limit_params=False)
    # print("num_layers, parameters_num, h_u_mean , h_u_exp_mean , grad_mean , exp_grad_mean , h_u_plus_exp_grad_mean")
    # num_layers_list = []
    # parameters_num_list = []
    # h_u_mean_list = []
    # h_u_exp_mean_list = []
    # grad_mean_list = []
    # exp_grad_mean_list = []
    # h_u_plus_exp_grad_mean_list = []
    #
    # for i in range(min_fc_layers, max_fc_layers):
    #     num_layers, parameters_num \
    #         , h_u_mean, h_u_exp_mean, grad_mean, exp_grad_mean, h_u_plus_exp_grad_mean \
    #         , exp_alpha_mean, exp_alpha_prod_mean, exp_alpha_frac_mean\
    #         = run_model(BernoulliDropoutNet, additional_fc_layers=i)
    #
    #     num_layers_list.append(num_layers)
    #     parameters_num_list.append(parameters_num)
    #     h_u_mean_list.append(h_u_mean)
    #     h_u_exp_mean_list.append(h_u_exp_mean)
    #     grad_mean_list.append(grad_mean)
    #     exp_grad_mean_list.append(exp_grad_mean)
    #     h_u_plus_exp_grad_mean_list.append(h_u_plus_exp_grad_mean)
    #
    # print("num_layers_bernoulli")
    # print(num_layers_list)
    # print("parameters_num_bernoulli")
    # print(parameters_num_list)
    # print("h_u_mean_bernoulli")
    # print(h_u_mean_list)
    # print("h_u_exp_mean_bernoulli")
    # print(h_u_exp_mean_list)
    # print("grad_mean_bernoulli")
    # print(grad_mean_list)
    # print("exp_grad_mean_bernoulli")
    # print(exp_grad_mean_list)
    # print("h_u_plus_exp_grad_mean_bernoulli")
    # print(h_u_plus_exp_grad_mean_list)
    #
    # bernoulli_results = [h_u_mean_list, h_u_exp_mean_list, grad_mean_list, exp_grad_mean_list, h_u_plus_exp_grad_mean_list]
    # compare_results = []
    # for i, gauss_list in enumerate(gauss_results):
    #     bernoulli_list = bernoulli_results[i]
    #     compare_list = []
    #
    #     for j, gauss_val in enumerate(gauss_list):
    #         bernoulli_val = bernoulli_list[j]
    #         compare_list.append(np.abs(gauss_val - bernoulli_val))
    #
    #     compare_results.append(compare_list)
    #
    # print("h_u_mean_diff")
    # print(compare_results[0])
    # print("h_u_exp_mean_diff")
    # print(compare_results[1])
    # print("grad_mean_diff")
    # print(compare_results[2])
    # print("exp_grad_mean_diff")
    # print(compare_results[3])
    # print("h_u_plus_exp_grad_mean_diff")
    # print(compare_results[4])
    #
    # y_list = compare_results
    # series_labels = ["f1=h_u", "f2=exp(h_u)-1", "f3=||grad_u(l)||^2", "f2*f3", "f1+2*f2*f3"]
    # title = "comparing Gaussian and Bernoulli Dropout, 2500 parameters per layer"
    # ylabel = "ampirical averages difference in absolute value"
    # xlabel = "number of layers"
    #
    # save_fig(x=num_layers_list, y_list=y_list, series_labels=series_labels, title=title, xlabel=xlabel, ylabel=ylabel)


def run_bernoulli_dropout_limited(gauss_results, min_fc_layers=0, max_fc_layers=args.max_fc_layers):
    run_bernoulli(gauss_results=gauss_results, min_fc_layers=min_fc_layers, max_fc_layers=max_fc_layers
                  , limit_params=True)

    # print("num_layers, parameters_num, h_u_mean , h_u_exp_mean , grad_mean , exp_grad_mean , h_u_plus_exp_grad_mean")
    # num_layers_list = []
    # parameters_num_list = []
    # h_u_mean_list = []
    # h_u_exp_mean_list = []
    # grad_mean_list = []
    # exp_grad_mean_list = []
    # h_u_plus_exp_grad_mean_list = []
    #
    # for i in range(min_fc_layers, max_fc_layers):
    #     num_layers, parameters_num \
    #         , h_u_mean, h_u_exp_mean, grad_mean, exp_grad_mean, h_u_plus_exp_grad_mean \
    #         , exp_alpha_mean, exp_alpha_prod_mean, exp_alpha_frac_mean\
    #         = run_model(BernoulliDropoutNetLimited, additional_fc_layers=i)
    #
    #     num_layers_list.append(num_layers)
    #     parameters_num_list.append(parameters_num)
    #     h_u_mean_list.append(h_u_mean)
    #     h_u_exp_mean_list.append(h_u_exp_mean)
    #     grad_mean_list.append(grad_mean)
    #     exp_grad_mean_list.append(exp_grad_mean)
    #     h_u_plus_exp_grad_mean_list.append(h_u_plus_exp_grad_mean)
    #
    # print("num_layers_bernoulli")
    # print(num_layers_list)
    # print("parameters_num_bernoulli")
    # print(parameters_num_list)
    # print("h_u_mean_bernoulli")
    # print(h_u_mean_list)
    # print("h_u_exp_mean_bernoulli")
    # print(h_u_exp_mean_list)
    # print("grad_mean_bernoulli")
    # print(grad_mean_list)
    # print("exp_grad_mean_bernoulli")
    # print(exp_grad_mean_list)
    # print("h_u_plus_exp_grad_mean_bernoulli")
    # print(h_u_plus_exp_grad_mean_list)
    #
    # paramrters_mean = np.sum(parameters_num_list)/len(parameters_num_list)
    #
    # bernoulli_results = [h_u_mean_list, h_u_exp_mean_list, grad_mean_list, exp_grad_mean_list, h_u_plus_exp_grad_mean_list]
    # compare_results = []
    # for i, gauss_list in enumerate(gauss_results):
    #     bernoulli_list = bernoulli_results[i]
    #     compare_list = []
    #
    #     for j, gauss_val in enumerate(gauss_list):
    #         bernoulli_val = bernoulli_list[j]
    #         compare_list.append(np.abs(gauss_val - bernoulli_val))
    #
    #     compare_results.append(compare_list)
    #
    # print("h_u_mean_diff")
    # print(compare_results[0])
    # print("h_u_exp_mean_diff")
    # print(compare_results[1])
    # print("grad_mean_diff")
    # print(compare_results[2])
    # print("exp_grad_mean_diff")
    # print(compare_results[3])
    # print("h_u_plus_exp_grad_mean_diff")
    # print(compare_results[4])
    #
    # y_list = compare_results
    # series_labels = ["f1=h_u", "f2=exp(h_u)-1", "f3=||grad_u(l)||^2", "f2*f3", "f1+2*f2*f3"]
    # title = "comparing Gaussian and Bernoulli Dropout, ~" + str(paramrters_mean) + " parameters total"
    # ylabel = "ampirical averages difference in absolute value"
    # xlabel = "number of layers"
    #
    # save_fig(x=num_layers_list, y_list=y_list, series_labels=series_labels, title=title, xlabel=xlabel, ylabel=ylabel)


########################################################################
start_time = datetime.now()

print("args")
print(args)

# if args.limit_params:
#     run_gauss_dropout_limited()
# else:
#     run_gauss_dropout()

# run_gauss()
# run_gauss(limit_params=True)
run_gauss_dropout(test_before_train=args.test_before_train, test_after_train=args.test_after_train)
run_gauss_dropout_limited(test_before_train=args.test_before_train, test_after_train=args.test_after_train)

end_time = datetime.now()
print("run time: " + str(end_time - start_time))
