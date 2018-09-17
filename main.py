from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Linear
from torch.nn import Parameter 
from torchvision import datasets, transforms
from torch.autograd.function import InplaceFunction
from torch.autograd.function import Function
from torch.nn import Module
import math
import numpy as np
import os, errno
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def calc_kl(model):
    kl = 0.5*((list(model.parameters())[1]**2).sum().data + (list(model.parameters())[0]**2).sum().data)/model.prior_var
    return kl.item()


def save_fig(x, y_list, series_labels, title, xlabel, ylabel, linewidths=None, linestyles=None, log_yscale=False, log_xscale=False):

    if linewidths is None:
        linewidths = [1 for y in y_list]
    if linestyles is None:
        linestyles = ['solid' for y in y_list]

    # markers_unique = ["x", "o", "v", "d", ",", "^", "*", "|"]
    markers_unique = ["x", "o", "d", "^", ","]

    markers_repeats = int(math.floor(len(y_list)/len(markers_unique))) + 1
    markers = markers_unique * markers_repeats
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for i, y in enumerate(y_list):
        label = series_labels[i]
        ax1.scatter(x, y, s=10, label=label, marker=markers[i])
        ax1.plot(x, y, linestyle=linestyles[i], linewidth=linewidths[i])
    plt.legend(loc='upper right')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if log_yscale:
        plt.yscale('log')
    if log_xscale:
        plt.xscale('log')
    plt.title(title)
    try:
        os.makedirs("saved_figures")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    plt.savefig("saved_figures/" + title + ".png")
    # plt.show()


def gdropout(input, p=0.5, training=False, inplace=False):
    return gDropout.apply(input, p, training, inplace)


def gdropout(input, p=0.5, training=False, inplace=False):
    return gDropout.apply(input, p, training, inplace)


class gDropout(InplaceFunction):

    @staticmethod
    def _make_noise(input):
        return input.new().resize_as_(input)

    @staticmethod
    def symbolic(g, input, p=0.5, train=False, inplace=False):
        # See Note [Export inplace]
        r, _ = g.op("Dropout", input, ratio_f=p, is_test_i=not train, outputs=2)
        return r

    @classmethod
    def forward(cls, ctx, input, p=0.5, train=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError("gdropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        ctx.p = p
        ctx.train = train
        ctx.inplace = inplace

        if ctx.p == 0 or not ctx.train:
            return input

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        ctx.noise = cls._make_noise(input)
        if ctx.p == 1:
            ctx.noise.fill_(0)
        else:
            #ctx.noise.bernoulli_(1 - ctx.p).div_(1 - ctx.p)
            sigma = p/(1-p)
            ctx.noise = 1 + sigma * torch.randn_like(ctx.noise)
        ctx.noise = ctx.noise.expand_as(input)
        output.mul_(ctx.noise)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.train:
            return grad_output * ctx.noise, None, None, None
        else:
            return grad_output, None, None, None


class gDropconnectLinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        p=0.5
        sigma = p/(1-p)
        ctx.noise = 1 + sigma * torch.randn_like(weight)
        ctx.save_for_backward(input, weight, bias)
        output = input.mm((weight * ctx.noise).t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight * ctx.noise)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input) * ctx.noise
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias


class gDropconnectLinear(Module):
    r"""
    """

    def __init__(self, in_features, out_features, bias=True):
        super(gDropconnectLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)
        self.weight.data.uniform_(1-stdv, 1+stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return gDropconnectLinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class gDropconnectLinearFunctionADD(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, sigma=1):
        ctx.noise = sigma * torch.randn_like(weight)
        ctx.save_for_backward(input, weight, bias)
        output = input.mm((weight + ctx.noise).t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight + ctx.noise)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, None


class gDropconnectLinearADD(Module):
    r"""
    """

    def __init__(self, in_features, out_features, bias=True, var=0.05):
        super(gDropconnectLinearADD, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.sigma = math.sqrt(var)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        #self.weight.data.uniform_(1-stdv, 1+stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return gDropconnectLinearFunctionADD.apply(input, self.weight, self.bias, self.sigma)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class gDropoutNet(nn.Module):
    def __init__(self):
        super(gDropoutNet, self).__init__()
        self.fc1 = nn.Linear(784, 600)
        self.fc2 = nn.Linear(600, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = F.gdropout(x, p=0.5, training=self.training)
        x = gdropout(x, p=0.5, training=self.training)
        x = F.relu(x)
        x = self.fc2(x)
        #x = F.gdropout(x, p=0.5, training=self.training)
        x = gdropout(x, p=0.5, training=self.training)
        return F.log_softmax(x, dim=1)


class gDropconnectNet(nn.Module):
    def __init__(self, prior_var=0.05):
        super(gDropconnectNet, self).__init__()
        self.fc1 = gDropconnectLinearADD(784, 600, bias=False, var=prior_var)
        self.fc2 = gDropconnectLinearADD(600, 2, bias=False, var=prior_var)
        self.layers = [self.fc1, self.fc2]
        self.prior_var = prior_var

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def auto_update_noise(self, activate=True):
        return

    def update_noise(self, activate=True):
        return


class BinaryClassMNIST(datasets.MNIST):
    def __getitem__(self, i):
        input, target = super(BinaryClassMNIST, self).__getitem__(i)
        binary_target = 0
        if target > 4:
            binary_target = 1
        return input, binary_target


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        #for params in model.parameters():
        #    loss = loss + args.weight_decay * ((params - 1) ** 2).sum() 

        loss.backward()
        optimizer.step()
        
        #for params in model.parameters():
        #    params.data += 0.1 * params.data.abs().lt(0.01).type(torch.FloatTensor) * params.data.sign() * 0.1        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tKL: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), calc_kl(model)))


def train_accuracy(args, model, device, train_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(train_loader.dataset)
    kl_val = calc_kl(model)
    print('\nTrain set: Accuracy: {}/{} ({:.0f}%)\tKL: {:.6f}\n'.format(
        correct, len(train_loader.dataset),
        accuracy, kl_val))
    return accuracy, kl_val


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    kl_val = calc_kl(model)
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\tKL: {:.6f}\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy, calc_kl(model)))
    return accuracy, kl_val


def run_model(args, model, train_loader, test_loader):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    epochs = range(1, args.epochs + 1)
    train_accuracy_per_epoch = []
    test_accuracy_per_epoch = []
    kl_per_epoch = []
    for epoch in epochs:
        train(args, model, args.device, train_loader, optimizer, epoch)
        train_accuracy_val, kl_val = train_accuracy(args, model, args.device, train_loader)
        test_accuracy_val, kl_val = test(args, model, args.device, test_loader)
        train_accuracy_per_epoch.append(train_accuracy_val)
        test_accuracy_per_epoch.append(test_accuracy_val)
        kl_per_epoch.append(kl_val)
    final_kl = calc_kl(model)
    print("final kl")
    print(final_kl)

    xlabel = "epochs"
    x_data = epochs

    y_list = [train_accuracy_per_epoch, test_accuracy_per_epoch]
    series_labels = ["train_accuracy", "test_accuracy"]
    ylabel = "accuracy"

    title = "accuracy per epoch"

    save_fig(x=x_data, y_list=y_list, series_labels=series_labels,
             title=title, xlabel=xlabel, ylabel=ylabel)

    y_list = [kl_per_epoch]
    series_labels = ["kl"]
    ylabel = "kl"

    title = "kl per epoch"

    save_fig(x=x_data, y_list=y_list, series_labels=series_labels,
             title=title, xlabel=xlabel, ylabel=ylabel)
    return final_kl


def optimize_prior(args, train_loader, test_loader):
    kl_per_prior = []
    for prior_var in args.prior_var_list:
        model = gDropconnectNet(prior_var=prior_var).to(args.device)
        kl = run_model(args, model, train_loader, test_loader)
        kl_per_prior.append(kl)
    min_kl_idx = np.argmin(kl_per_prior)
    min_kl_prior = args.prior_var_list[min_kl_idx]
    min_kl = kl_per_prior[min_kl_idx]
    print("min_kl_prior")
    print(min_kl_prior)
    print("min_kl")
    print(min_kl)

    y_list = [kl_per_prior]
    series_labels = ["kl"]
    ylabel = "kl"

    xlabel = "prior_var"

    title = "kl per prior"

    save_fig(x=args.prior_var_list, y_list=y_list, series_labels=series_labels,
             title=title, xlabel=xlabel, ylabel=ylabel, log_yscale=True, log_xscale=True)

    return min_kl_prior, min_kl


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--optimize-prior-var', action='store_true', default=False,
                        help='optimize prior var')
    parser.add_argument('-prior_var_list', type=list,
                        default=[0.1 * np.exp(-j) for j in range(11)],
                        metavar='N', help='list of epochs for each part in the training')

    args = parser.parse_args()
    args.no_cuda = True
    args.seed = 2344
    args.batch_size = 64
    args.test_batch_size = 1000
    args.momentum = 0.5
    args.lr = 0.01
    args.log_interval = 500
    args.prior_var = 0.05
    args.optimize_prior_var = False
    args.prior_var_list = [0.1 * np.exp(-j) for j in range(11)]
    args.weight_decay = 0.005
    #args.weight_decay = 0.05
    args.epochs = 50
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    args.device = device
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    train_loader = torch.utils.data.DataLoader(BinaryClassMNIST('../data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(BinaryClassMNIST('../data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),batch_size=args.test_batch_size, shuffle=True, **kwargs)

    prior_var = args.prior_var
    if args.optimize_prior_var:
        min_kl_prior, min_kl = optimize_prior(args, train_loader, test_loader)
        prior_var = min_kl_prior
    model = gDropconnectNet(prior_var=prior_var).to(args.device)
    run_model(args, model, train_loader, test_loader)


if __name__ == '__main__':
    main()
