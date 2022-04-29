import argparse
import os
from pickletools import optimize
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

from vocab import Vocab
from model import *
from utils import *
from batchify import get_batches
from train import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', metavar='DIR', required=True,
                    help='checkpoint directory')
parser.add_argument('--output', metavar='FILE',
                    help='output file name (in checkpoint directory)')
parser.add_argument('--data', metavar='FILE',
                    help='path to data file')

parser.add_argument('--enc', default='mu', metavar='M',
                    choices=['mu', 'z'],
                    help='encode to mean of q(z|x) or sample z from q(z|x)')
parser.add_argument('--dec', default='greedy', metavar='M',
                    choices=['greedy', 'sample'],
                    help='decoding algorithm')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='batch size')
parser.add_argument('--max-len', type=int, default=35, metavar='N',
                    help='max sequence length')

parser.add_argument('--seed', type=int, default=1111, metavar='N',
                    help='random seed')
parser.add_argument('--no-cuda', action='store_true',
                    help='disable CUDA')

def get_models(path):
    ckpt = torch.load(os.path.join(path, 'model.pt'))
    train_args = ckpt['args']
    model = {'dae': DAE, 'vae': VAE, 'aae': AAE}[train_args.model_type](
        vocab, train_args).to(device)

    model.load_state_dict(ckpt['model'])
    model.flatten()
    model.eval()

    ckpt = torch.load(os.path.join(path, 'classifier.pt'))
    classifier = Classifier(train_args).to(device)
    classifier.load_state_dict(ckpt['model'])
    classifier.eval()

    return model, classifier

def encode(sents):
    assert args.enc == 'mu' or args.enc == 'z'
    batches, order = get_batches(sents, vocab, args.batch_size, device)
    z = []
    s = []
    for inputs, _, target_style in batches:
        mu, logvar = model.encode(inputs)
        if args.enc == 'mu':
            zi = mu
        else:
            zi = reparameterize(mu, logvar)
        z.append(zi.detach().cpu().numpy())
        s.append(target_style.T.detach().cpu().numpy())

    z = np.concatenate(z, axis=0)
    z_ = np.zeros_like(z)
    z_[np.array(order)] = z

    s = np.concatenate(s, axis=0)
    s_ = np.zeros_like(s)
    s_[np.array(order)] = s

    return z_, s_, order


def get_z_prime(z, s, order):
    loss_func = torch.nn.BCELoss()
    z_prime = []

    for i in range(len(z)):
        print("Optimizing", i)
        z_ = torch.tensor(z[i]).to(device)
        s_ = torch.tensor(s[i]).to(device)

        z_.requires_grad_(True)
        optimizer = optim.Adam([z_], lr=0.01)

        while True:
            y = classifier.forward(z_) 
            loss = loss_func(y, s_.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(abs((y-s_.float()).item()), loss)

            if loss.item() < 0.1:
                break

        z[i] = z_.detach().cpu().numpy()

    # z_prime = np.concatenate(z_prime, axis=0)
    # z_ = np.zeros_like(z_prime)
    # z_[np.array(order)] = z_prime

    return z


def decode(z):
    sents = []
    i = 0
    while i < len(z):
        zi = torch.tensor(z[i: i+args.batch_size], device=device)
        outputs = model.generate(zi, args.max_len, args.dec).t()
        for s in outputs:
            sents.append([vocab.idx2word[id] for id in s[1:]])  # skip <go>
        i += args.batch_size
    return strip_eos(sents)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'

    args = parser.parse_args()
    vocab = Vocab(os.path.join(args.checkpoint, 'vocab.txt'))
    set_seed(args.seed)
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model, classifier = get_models(args.checkpoint)

    sents = load_sent(args.data)
    z, s, order = encode(sents)

    z = get_z_prime(z, s, order)

    sents_rec = decode(z)
    write_z(z, os.path.join(args.checkpoint, args.output+'.z'))
    write_sent(sents_rec, os.path.join(args.checkpoint, args.output+'.rec'))

