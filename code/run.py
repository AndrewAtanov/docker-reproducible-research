import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from sklearn import datasets
import utils
import datautils
import argparse
from models.distributions import GMM
from models.realnvp import RealNVP, NFGMM
from scipy import linalg
import matplotlib as mpl

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=1000, type=int)
parser.add_argument('--train_size', default=3000, type=int)
parser.add_argument('--test_size', default=1000, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--k_prior', default=2, type=int)
args = parser.parse_args()

sns.set(style='whitegrid')
colors = np.array(sns.color_palette(n_colors=10))


def draw_ellipse(covar, mean, color, splot, alpha=0.5, label=''):
    if len(covar.shape) == 1:
        covar = np.diag(covar)
    v, w = linalg.eigh(covar)
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180. * angle / np.pi  # convert to degrees
    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
    ell.set_alpha(alpha)
    splot.add_patch(ell)
    ell.set(label=label)


def plot_gmm(gmm, colors, ax):
    pi = utils.tonp(torch.softmax(gmm.weights, 0))
    for i, [covar, mean, w, c] in enumerate(zip(gmm.covariances, gmm.means, pi, colors[1:])):
        draw_ellipse(utils.tonp(covar), utils.tonp(mean), c, ax,
                     alpha=w, label=r'$\pi_{}$ = {:.3f}'.format(i+1, w))


def plot(title='', data=None):
    x, y = map(utils.tonp, [x_test, y_test])
    z = flow.f(torch.FloatTensor(x).to(d))[0].detach().cpu().numpy()

    plt.figure(figsize=(12, 12))
    for t, c in zip(np.unique(y), colors):
        idxs = (y == t)
        plt.subplot(2, 2, 1)
        plt.title('Z space (Data map)')
        plt.scatter(z[idxs, 0], z[idxs, 1], c=[c], alpha=.6)
        plt.subplot(2, 2, 2)
        plt.title('X space (Data)')
        plt.scatter(x[idxs, 0], x[idxs, 1], c=[c], alpha=.6)

    z, y = flow.prior.sample((10000,), labels=True)
    x = flow.g(z).detach().cpu().numpy()
    splot = plt.subplot(2, 2, 3)
    plt.title('Z space (Prior samples)')
    z = utils.tonp(z)
    plt.scatter(z[:, 0], z[:, 1], s=5, c=colors[0], alpha=.0)
    pi = utils.tonp(flow.prior.get_weights())
    arg = np.arange(args.k_prior)
    covs = utils.tonp(flow.prior.covariances)
    mu = utils.tonp(torch.stack([m for m in flow.prior.means]))
    for i, [covar, mean, w, c] in enumerate(zip(covs[arg[:10]],
                                                mu[arg[:10]],
                                                pi[arg[:10]], colors[1:])):
        draw_ellipse(covar, mean, c, splot, alpha=w, label=r'$\pi_{}$ = {:.3f}'.format(i+1, w))
    splot.legend()

    splot = plt.subplot(2, 2, 4)
    plt.title('X space (Flow samples)')
    plt.scatter(x[:, 0], x[:, 1], s=5, alpha=.2, c=colors[1:][y.clip(0, 9)])
    plt.suptitle(title, size=25)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

x_train, y_train = datasets.make_moons(n_samples=args.train_size, noise=0.1, random_state=args.seed)
x_test, y_test = datasets.make_moons(n_samples=args.test_size, noise=0.1, random_state=args.seed)
x_train, x_test = map(torch.FloatTensor, [x_train, x_test])


d = torch.device('cpu')

nets = lambda: nn.Sequential(nn.Linear(2, 32),
                             nn.LeakyReLU(),
                             nn.Linear(32, 2),
                             nn.Tanh())

nett = lambda: nn.Sequential(nn.Linear(2, 32),
                             nn.LeakyReLU(),
                             nn.Linear(32, 2))

masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 2).astype(np.float32))

covars = torch.rand(args.k_prior, 2, 2)
covars = torch.matmul(covars, covars.transpose(1, 2))
covars = torch.stack([torch.eye(2)] * args.k_prior)
prior = GMM(means=torch.randn(args.k_prior, 2) * 2,
            covariances=covars,
            weights=torch.FloatTensor([1. / args.k_prior] * args.k_prior),
            normalize=False)
flow = NFGMM(nets, nett, masks, prior).to(d)

gamma = torch.zeros(args.train_size, args.k_prior)

params = [p for p in flow.t.parameters()] + [p for p in flow.s.parameters()]
opt = torch.optim.Adam(params)

train, test = [], []

for epoch in range(1, args.epochs + 1):
    # E-step
    z, logdet = flow.f(x_train)
    logp = torch.stack([flow.prior.log_prob(z, k=k) + logdet for k in range(args.k_prior)])
    gamma.data = torch.exp(logp - torch.logsumexp(logp, dim=0)).transpose(0, 1)

    # M-step
    # prior
    n = gamma.sum(0) + 1e-8
    mean = torch.sum(gamma[..., None] * z[:, None], dim=0) / n[:, None]
    covs = (z[:, None] - mean[None])[..., None]
    covs = torch.matmul(covs, covs.transpose(2, 3)) * gamma[..., None, None]
    covs = covs.sum(0) / n[:, None, None] + torch.eye(2)[None].to(d) * 1e-8

    flow.prior.weights.data = n.to(d) / args.train_size
    for i in range(args.k_prior):
        flow.prior.set_covariance(i, covs[i])
        flow.prior.means[i].data = mean[i]

    # \theta
    loss = -(logp.transpose(0, 1) * gamma).sum()
    opt.zero_grad()
    loss.backward()
    opt.step()

    train_logp = flow.log_prob(x_train).mean().item()
    test_logp = flow.log_prob(x_test).mean().item()
    train.append(train_logp)
    test.append(test_logp)

    if epoch % 10 == 0:
        print('epoch %s:' % epoch, ' | train logp %.3f' % train_logp, ' | test logp %.3f' % test_logp)

plot()
plt.savefig('../latex/figs/plot_{}.png'.format(args.k_prior), dpi=300, bbox_inches='tight')
plt.clf()

plt.xlabel('epoch', size=20)
plt.ylabel(r'$p_{flow}$', size=20)
plt.plot(train, label='Train', lw=5, alpha=.7)
plt.plot(test, label='Test', lw=5, alpha=.7)
plt.legend()
plt.savefig('../latex/figs/training_curves.png', dpi=200, bbox_inches='tight')
