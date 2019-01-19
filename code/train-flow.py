from models import realnvp
from models import distributions
import torch
import utils
import datautils
from sklearn.mixture import GaussianMixture
import os
from logger import Logger
import time
import numpy as np
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--name', default='')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--data', default='blobs')
parser.add_argument('--data_cmplx', default=1, type=int)
parser.add_argument('--data_save', action='store_true')
parser.add_argument('--train_bs', default=32, type=int)
parser.add_argument('--test_bs', default=512, type=int)
parser.add_argument('--num_examples', default=100, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--lr_schedule', default='no')
parser.add_argument('--model', default='')
parser.add_argument('--prior_train_algo', default='GD')
parser.add_argument('--prior', default='gmm')
parser.add_argument('--gmm_k', default=1, type=int)
parser.add_argument('--gmm_init', default='random')
parser.add_argument('--gmm_cov', default='full')
parser.add_argument('--log_each', default=10, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--data_seed', default=0, type=int)
args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fmt = {
    'time': '.3f'
}
logger = Logger('logs', base='../logs/', fmt=fmt)

# Load data
np.random.seed(args.data_seed)
torch.manual_seed(args.data_seed)
torch.cuda.manual_seed_all(args.data_seed)
trainloader, testloader, data_shape = datautils.load_dataset(args.data, args.train_bs, args.test_bs,
                                                             num_examples=args.num_examples,
                                                             seed=args.data_seed,
                                                             complexity=args.data_cmplx)

if args.data_save:
    torch.save(testloader, '../data/testloader.pckl')
    torch.save(trainloader, '../data/trainloader.pckl')

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
# Create model
if args.prior == 'gmm':
    if args.gmm_k == 1 and args.prior_train_algo == 'no':
        prior = torch.distributions.MultivariateNormal(torch.zeros((2,)), torch.eye(2))
    else:
        prior = distributions.GMM(k=args.gmm_k, dim=2, normalize=args.prior_train_algo == 'GD')
else:
    raise NotImplementedError

if args.model == 'toy':
    flow = realnvp.get_toy_nvp(prior=prior, device=device)
else:
    raise NotImplementedError

if args.gmm_init == 'GMM':
    z = utils.batch_eval(lambda x: utils.tonp(flow.f(x[0].to(device))[0]), trainloader)
    z = np.concatenate(z)

    gmm = GaussianMixture(n_components=args.gmm_k, covariance_type=args.gmm_cov).fit(z)
    for i in range(args.gmm_k):
        flow.prior.means[i].data = torch.FloatTensor(gmm.means_[i]).to(device)
        flow.prior.set_covariance(i, torch.FloatTensor(gmm.covariances_[i]).to(device))
    flow.prior.weights.data = torch.FloatTensor(np.log(gmm.weights_)).to(device)

torch.save(flow.state_dict(), os.path.join('../data/', 'model.torch'))

if args.gmm_k == 1:
    pass
elif args.prior_train_algo == 'no':
    for p in flow.prior.parameters():
        p.requires_grad = False
elif args.prior_train_algo not in ['GD', 'no']:
    raise NotImplementedError

optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad], lr=args.lr)
if args.lr_schedule == 'no':
    lr_scheduler = utils.BaseLR(optimizer)
elif args.lr_schedule == 'linear':
    lr_scheduler = utils.LinearLR(optimizer, args.epochs)
else:
    raise NotImplementedError

t0 = time.time()
for epoch in range(1, args.epochs + 1):
    train_logp = 0.
    for x, _ in trainloader:
        x = torch.FloatTensor(x).to(device)
        loss = -flow.log_prob(x).mean()
        train_logp += -loss.item() * x.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_logp /= len(trainloader.dataset)
    test_logp = torch.cat(utils.batch_eval(lambda x: flow.log_prob(x[0].to(device)), testloader)).mean().item()
    lr_scheduler.step()

    if epoch % args.log_each == 0 or epoch == 1:
        logger.add_scalar(epoch, 'train.logp', train_logp)
        logger.add_scalar(epoch, 'test.logp', test_logp)
        logger.add_scalar(epoch, 'time', time.time() - t0)
        t0 = time.time()
        logger.iter_info()
        logger.save()

        torch.save(flow.state_dict(), os.path.join('../', 'model.torch'))

parser.done()
