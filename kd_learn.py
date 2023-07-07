import rnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import decomposition
from scipy import linalg
import copy

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



def compute_loss(model, device, test_loader, is_print=True, topk=[1], features=None):
    '''
    Function that computes the top-k accuracy of model for dataset=test_loader
    :param nn.Module model: reduced net
    :param torch.device device: object representing the device on
        which a torch.Tensor is or will be allocated.
    :param iterable test_loader: iterable object, it load the dataset.
        It iterates over the given dataset, obtained combining a
        dataset(images and labels) and a sampler.
    :param bool is_print:
    :param list top_k
    :return float test_accuracy
    '''
    model.eval()
    model.to(device)
    test_loss = 0

    res = []
    maxk = max(topk)
    batch_old = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            batch = data.size()[0]
            if features is None:
                output = model(data)
            else:
                output = model(data, features=features[batch_old : batch_old + batch, :])
            batch_old = batch
            test_loss += F.nll_loss(output, target,
                                    reduction='sum').item()  # sum up batch loss
            # torch.tok Returns the k largest elements of the given
            # input tensor along a given dimension.
            _, pred = torch.topk(output, maxk, dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k)
    test_loss /= len(test_loader.sampler)
    correct = torch.FloatTensor(res).view(-1, len(topk)).sum(dim=0)
    test_accuracy = 100. * correct / len(test_loader.sampler)
    for idx, k in enumerate(topk):
        print(' Top {}:  Accuracy: {}/{} ({:.2f}%)'.format(
            k, correct[idx], len(test_loader.sampler), test_accuracy[idx]))
        print('Loss Value:', test_loss)
    if len(topk) == 1:
        return test_accuracy[0]
    else:
        return test_accuracy


def train_kd(student,
        teacher,
        device,
        train_loader,
        optimizer,
        train_max_batch,
        alpha=0.0,
        temperature=1.,
        lr_decrease=None,
        epoch=1,
        features=None):
    '''
    Function that retrains the model with knowledge distillation
    when alpha=0, it reduced to the original training
    :param nn.Module student: reduced net
    :param nn.Module teacher: full net
    :param torch.device device: object representing the device on
        which a torch.Tensor is or will be allocated.
    :param iterable train_loader: iterable object, it load the dataset for
        training. It iterates over the given dataset, obtained combining a
        dataset(images and labels) and a sampler.
    :param optimizer
    :param train_max_batch
    :param float alpha: regularization parameter. Default value set to 0.0,
        i.e. when the training is reduced to the original one
    :param float temperature: temperature factor introduced. When T tends to
        infinity all the classes have the same probability, whereas when T
        tends to 0 the targets become one-hot labels. Default value set to 1.
    :param lr_decrease:
    :param int epoch: epoch number
    :return: accuracy
    '''
    student.train()
    teacher.eval()
    student.to(device)
    teacher.to(device)
    correct = 0.0
    batch_old = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        batch = data.size()[0]
        optimizer.zero_grad()
        if features is None:
            output = student(data)
        else:
            output = student(data, features[batch_old : batch_old + batch, :])
        output_teacher = teacher(data)
        batch_old = batch

        # The Kullback-Leibler divergence loss measure
        loss = nn.KLDivLoss()(F.log_softmax(output / temperature, dim=1),F.softmax(output_teacher / temperature, dim=1)
                             )*(alpha*temperature*temperature) + \
                 F.cross_entropy(output, target) * (1. - alpha)

        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum().item()
    print('Train Loss kd:', loss.item() / len(train_loader.sampler))
    train_loss_val = loss.item() / len(train_loader.sampler)
    accuracy = correct / len(train_loader.sampler) * 100.0
    if lr_decrease is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decrease
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * (epoch) / (epoch + 1)
    return accuracy, train_loss_val