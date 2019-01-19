import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
import numpy as np


class RealNVP(nn.Module):
    def __init__(self, nets, nett, masks, prior):
        super(RealNVP, self).__init__()

        self.prior = prior
        self.mask = nn.Parameter(masks, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(masks))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(masks))])

    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self, x):
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.g(z)
        return x


class NFGMM(RealNVP):
    def log_prob(self, x, k=None):
        if k is None:
            z, logp = self.f(x)
            return self.prior.log_prob(z) + logp
        else:
            z, logp = self.f(x)
            return self.prior.log_prob(z, k=k) + logp


def gmm_prior(k):
    covars = torch.rand(args.gmm_k, 2, 2)
    covars = torch.matmul(covars, covars.transpose(1, 2))
    prior = distributions.GMM(torch.randn(args.gmm_k, 2), covars, torch.FloatTensor([0.5] * args.gmm_k),
                              normalize=args.prior_train_algo == 'GD')


def get_toy_nvp(prior=None, device=None):
    def nets():
        return nn.Sequential(nn.Linear(2, 256),
                             nn.LeakyReLU(),
                             nn.Linear(256, 256),
                             nn.LeakyReLU(),
                             nn.Linear(256, 2),
                             nn.Tanh()
                             )

    def nett():
        return nn.Sequential(nn.Linear(2, 256),
                             nn.LeakyReLU(),
                             nn.Linear(256, 256),
                             nn.LeakyReLU(),
                             nn.Linear(256, 2)
                             )

    if prior is None:
        prior = distributions.MultivariateNormal(torch.zeros(2).to(device),
                                                 torch.eye(2).to(device))

    masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))
    return RealNVP(nets, nett, masks, prior).to(device)
