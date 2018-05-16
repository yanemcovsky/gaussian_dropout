# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import gauss_var, fix_params_layer_size
from dataloaders import args
from torch.autograd import Variable


# Conv2d with gaussian dropout on connect
class GDConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, activate_noise=True):
        super(GDConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias)
        self.p = 0.5
        self.var = gauss_var(self.p)
        self.mean = torch.zeros_like(self.weight) + 1
        self.noise = torch.normal(self.mean, self.var)
        self.activate_noise = activate_noise
        print("using multiplicative gaussian dropout in convolution layer with mean=1 (tensor) and var="
              + str(self.var))

    def forward(self, input):
        noisy_weight = self.weight
        if self.activate_noise:
            noisy_weight = self.weight * self.noise
        return F.conv2d(input, noisy_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def update_noise(self, activate=True):
        self.activate_noise = activate
        if activate:
            self.noise = torch.normal(self.mean, self.var)

    def noise_flat(self):
        noise_flat = self.noise.view(self.noise.numel())
        return noise_flat


# Linear with gaussian dropout on connect
class GDLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, active_noise=True):
        super(GDLinear, self).__init__(in_features, out_features, bias)

        self.p = 0.5
        self.var = gauss_var(self.p)
        self.mean = torch.zeros_like(self.weight) + 1
        self.noise = torch.normal(self.mean, self.var)
        self.active_noise = active_noise
        print("using multiplicative gaussian dropout in linear layer with mean=1 (tensor) and var="
              + str(self.var))

    def forward(self, input):
        noisy_weight = self.weight
        if self.active_noise:
            noisy_weight = self.weight * self.noise
        return F.linear(input, noisy_weight, self.bias)

    def update_noise(self, activate=True):
        self.active_noise = activate
        if activate:
            self.noise = torch.normal(self.mean, self.var)

    def noise_flat(self):
        noise_flat = self.noise.view(self.noise.numel())
        return noise_flat


# Conv2d with bernoulli dropout on connect
class BDConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, activate_noise=True):
        super(BDConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias)
        self.p = 0.5
        self.noise = torch.bernoulli(torch.zeros_like(self.weight) + self.p)
        self.activate_noise = activate_noise
        print("using multiplicative Bernoulli dropout in convolution layer with p=" + str(self.p))

    def forward(self, input):
        noisy_weight = self.weight
        if self.activate_noise:
            noisy_weight = self.weight * self.noise
        return F.conv2d(input, noisy_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def update_noise(self, activate=True):
        self.activate_noise = activate
        if activate:
            self.noise = torch.bernoulli(torch.zeros_like(self.weight) + self.p)

    def noise_flat(self):
        noise_flat = self.noise.view(self.noise.numel())
        return noise_flat


# Linear with bernoulli dropout on connect
class BDLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, active_noise=True):
        super(BDLinear, self).__init__(in_features, out_features, bias)

        self.p = 0.5
        self.noise = torch.bernoulli(torch.zeros_like(self.weight) + self.p)
        self.active_noise = active_noise
        print("using multiplicative Bernoulli dropout in linear layer with p=" + str(self.p))

    def forward(self, input):
        noisy_weight = self.weight
        if self.active_noise:
            noisy_weight = self.weight * self.noise
        return F.linear(input, noisy_weight, self.bias)

    def update_noise(self, activate=True):
        self.active_noise = activate
        if activate:
            self.noise = torch.bernoulli(torch.zeros_like(self.weight) + self.p)

    def noise_flat(self):
        noise_flat = self.noise.view(self.noise.numel())
        return noise_flat


class GaussDropoutNet(nn.Module):
    def __init__(self, additional_fc_layers=0, active_noise=True):
        super(GaussDropoutNet, self).__init__()

        self.p = 0.8
        self.var = gauss_var(self.p)
        self.mean = Variable(torch.zeros(args.batch_size, 1, 28, 28) + 1)
        self.noise = torch.normal(self.mean, self.var)
        self.active_noise = active_noise
        print("using multiplicative gaussian dropout in input layer with mean=1 (tensor) and var="
              + str(self.var))

        self.conv1 = GDConv2d(1, 10, kernel_size=5)
        self.conv2 = GDConv2d(10, 20, kernel_size=5)
        self.fc_first = GDLinear(320, 50)

        self.fc_additional = nn.ModuleList([GDLinear(50, 50) for i in range(additional_fc_layers)])
        self.fc_last = GDLinear(50, 10)

    def forward(self, x):
        if self.active_noise:
            x = x * self.noise

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc_first(x))
        for fc in self.fc_additional:
            x = F.relu(fc(x))
        x = self.fc_last(x)
        return F.log_softmax(x, dim=1)

    def update_noise(self, activate=True):
        self.active_noise = activate
        if activate:
            self.noise = torch.normal(self.mean, self.var)

        self.conv1.update_noise(activate)
        self.conv2.update_noise(activate)
        self.fc_first.update_noise(activate)
        for fc in self.fc_additional:
            fc.update_noise(activate)
        self.fc_last.update_noise(activate)

    def noise_flat(self):
        noise_flat = self.noise.view(self.noise.numel())
        noise_flat += self.conv1.noise_flat()
        noise_flat += self.conv2.noise_flat()
        noise_flat += self.fc_first.noise_flat()
        for fc in self.fc_additional:
            noise_flat += fc.noise_flat()
        noise_flat += self.fc_last.noise_flat()
        return noise_flat

    def train(self, mode=True):
        self.update_noise(mode)
        super(GaussDropoutNet, self).train()


class GaussDropoutNetLimited(nn.Module):
    def __init__(self, additional_fc_layers=0, active_noise=True):
        super(GaussDropoutNetLimited, self).__init__()

        self.p = 0.8
        self.var = gauss_var(self.p)
        self.mean = Variable(torch.zeros(args.batch_size, 1, 28, 28) + 1)
        self.noise = torch.normal(self.mean, self.var)
        self.active_noise = active_noise
        add_layers_size = fix_params_layer_size(additional_fc_layers)
        print("using multiplicative gaussian dropout in input layer with mean=1 (tensor) and var="
              + str(self.var) + " (scalar)")

        self.conv1 = GDConv2d(1, 10, kernel_size=5)
        self.conv2 = GDConv2d(10, 20, kernel_size=5)
        self.fc_first = GDLinear(320, add_layers_size)

        self.fc_additional = nn.ModuleList(
            [GDLinear(add_layers_size, add_layers_size) for i in range(additional_fc_layers)])
        self.fc_last = GDLinear(add_layers_size, 10)

    def forward(self, x):
        #
        if self.active_noise:
            x = x * self.noise

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc_first(x))
        for fc in self.fc_additional:
            x = F.relu(fc(x))
        x = self.fc_last(x)
        return F.log_softmax(x, dim=1)

    def update_noise(self, activate=True):
        self.active_noise = activate
        if activate:
            self.noise = torch.normal(self.mean, self.var)

        self.conv1.update_noise(activate)
        self.conv2.update_noise(activate)
        self.fc_first.update_noise(activate)
        for fc in self.fc_additional:
            fc.update_noise(activate)
        self.fc_last.update_noise(activate)

    def noise_flat(self):
        noise_flat = self.noise.view(self.noise.numel())
        noise_flat += self.conv1.noise_flat()
        noise_flat += self.conv2.noise_flat()
        noise_flat += self.fc_first.noise_flat()
        for fc in self.fc_additional:
            noise_flat += fc.noise_flat()
        noise_flat += self.fc_last.noise_flat()
        return noise_flat

    def train(self, mode=True):
        self.update_noise(mode)
        super(GaussDropoutNetLimited, self).train()


class BernoulliDropoutNet(nn.Module):
    def __init__(self, additional_fc_layers=0, active_noise=True):
        super(BernoulliDropoutNet, self).__init__()

        self.p = 0.8
        self.noise = torch.bernoulli(Variable(torch.zeros(args.batch_size, 1, 28, 28) + self.p))
        self.active_noise = active_noise
        print("using multiplicative Bernoulli dropout in input layer with p=" + str(self.p))

        self.conv1 = BDConv2d(1, 10, kernel_size=5)
        self.conv2 = BDConv2d(10, 20, kernel_size=5)
        self.fc_first = BDLinear(320, 50)

        self.fc_additional = nn.ModuleList([BDLinear(50, 50) for i in range(additional_fc_layers)])
        self.fc_last = BDLinear(50, 10)

    def forward(self, x):
        #
        if self.active_noise:
            x = x * self.noise

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc_first(x))
        for fc in self.fc_additional:
            x = F.relu(fc(x))
        x = self.fc_last(x)
        return F.log_softmax(x, dim=1)

    def update_noise(self, activate=True):
        self.active_noise = activate
        if activate:
            self.noise = torch.bernoulli(torch.zeros_like(self.noise) + self.p)

        self.conv1.update_noise(activate)
        self.conv2.update_noise(activate)
        self.fc_first.update_noise(activate)
        for fc in self.fc_additional:
            fc.update_noise(activate)
        self.fc_last.update_noise(activate)

    def noise_flat(self):
        noise_flat = self.noise.view(self.noise.numel())
        noise_flat += self.conv1.noise_flat()
        noise_flat += self.conv2.noise_flat()
        noise_flat += self.fc_first.noise_flat()
        for fc in self.fc_additional:
            noise_flat += fc.noise_flat()
        noise_flat += self.fc_last.noise_flat()
        return noise_flat

    def train(self, mode=True):
        self.update_noise(mode)
        super(BernoulliDropoutNet, self).train()


class BernoulliDropoutNetLimited(nn.Module):
    def __init__(self, additional_fc_layers=0, active_noise=True):
        super(BernoulliDropoutNetLimited, self).__init__()

        self.p = 0.8
        self.noise = torch.bernoulli(Variable(torch.zeros(args.batch_size, 1, 28, 28) + self.p))
        self.active_noise = active_noise
        add_layers_size = fix_params_layer_size(additional_fc_layers)
        print("using multiplicative Bernoulli dropout in input layer with p=" + str(self.p))

        self.conv1 = BDConv2d(1, 10, kernel_size=5)
        self.conv2 = BDConv2d(10, 20, kernel_size=5)
        self.fc_first = BDLinear(320, add_layers_size)

        self.fc_additional = nn.ModuleList(
            [BDLinear(add_layers_size, add_layers_size) for i in range(additional_fc_layers)])
        self.fc_last = BDLinear(add_layers_size, 10)

    def forward(self, x):
        if self.active_noise:
            x = x * self.noise

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc_first(x))
        for fc in self.fc_additional:
            x = F.relu(fc(x))
        x = self.fc_last(x)
        return F.log_softmax(x, dim=1)

    def update_noise(self, activate=True):
        self.active_noise = activate
        if activate:
            self.noise = torch.bernoulli(torch.zeros_like(self.noise) + self.p)

        self.conv1.update_noise(activate)
        self.conv2.update_noise(activate)
        self.fc_first.update_noise(activate)
        for fc in self.fc_additional:
            fc.update_noise(activate)
        self.fc_last.update_noise(activate)

    def noise_flat(self):
        noise_flat = self.noise.view(self.noise.numel())
        noise_flat += self.conv1.noise_flat()
        noise_flat += self.conv2.noise_flat()
        noise_flat += self.fc_first.noise_flat()
        for fc in self.fc_additional:
            noise_flat += fc.noise_flat()
        noise_flat += self.fc_last.noise_flat()
        return noise_flat

    def train(self, mode=True):
        self.update_noise(mode)
        super(BernoulliDropoutNetLimited, self).train()





