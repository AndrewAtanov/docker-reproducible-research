import numpy as np
import torch
from sklearn import datasets
import os
import torchvision
from torchvision import transforms
from models.distributions import GMM
import torch.nn.functional as F


class MyPad(object):
    def __init__(self, size, mode='reflect'):
        self.mode = mode
        self.size = size
        self.topil = transforms.ToPILImage()

    def __call__(self, img):
        return self.topil(pad(img, self.size, self.mode))


def load_dataset(data, train_bs, test_bs, num_examples=None, shuffle=True, seed=42, complexity=1):
    transform_train = transforms.Compose([
        MyPad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    if data == 'blobs':
        x, y = datasets.make_blobs(n_samples=int(num_examples * 1.4), random_state=seed)
        trainset = torch.utils.data.TensorDataset(torch.FloatTensor(x[:num_examples]),
                                                  torch.FloatTensor(y[:num_examples]))
        testset = torch.utils.data.TensorDataset(torch.FloatTensor(x[num_examples:]),
                                                 torch.FloatTensor(y[num_examples:]))
        data_shape = (2,)
    elif data == 'moons':
        x, y = datasets.make_moons(n_samples=int(num_examples * 1.4), noise=0.1, random_state=seed)
        trainset = torch.utils.data.TensorDataset(torch.FloatTensor(x[:num_examples]),
                                                  torch.FloatTensor(y[:num_examples]))
        testset = torch.utils.data.TensorDataset(torch.FloatTensor(x[num_examples:]),
                                                 torch.FloatTensor(y[num_examples:]))
        data_shape = (2,)
    elif data == 'circles':
        x, y = datasets.make_circles(n_samples=int(num_examples * 1.4), noise=0.1, factor=0.2, random_state=seed)
        trainset = torch.utils.data.TensorDataset(torch.FloatTensor(x[:num_examples]),
                                                  torch.FloatTensor(y[:num_examples]))
        testset = torch.utils.data.TensorDataset(torch.FloatTensor(x[num_examples:]),
                                                 torch.FloatTensor(y[num_examples:]))
        data_shape = (2,)
    elif data == 'gmm':
        gmm = random_gmm(k=complexity)
        x, y = gmm.sample((num_examples,), labels=True)
        trainset = torch.utils.data.TensorDataset(x.detach(), torch.LongTensor(y))
        trainset.data_model = gmm
        x, y = gmm.sample((int(num_examples * 0.4),), labels=True)
        testset = torch.utils.data.TensorDataset(x.detach(), torch.LongTensor(y))
        data_shape = (2,)
    else:
        raise NotImplementedError

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=shuffle, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False, num_workers=0)

    return trainloader, testloader, data_shape


class SampleLoader(object):
    def __init__(self, feeder, n=1):
        self.n = n
        self.feeder = feeder
        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter < self.n:
            self.counter += 1
            return self.feeder()
        else:
            self.counter = 0
            raise StopIteration


def random_gmm(k, dim=2):
    cov = torch.rand(k, dim, dim) - 0.5
    cov = torch.matmul(cov, cov.transpose(1, 2))
    mean = (torch.rand(k, dim) - 0.5) * 10
    weight = F.softmax(torch.rand(k), dim=0)
    return GMM(means=mean, covariances=cov, weights=weight)
