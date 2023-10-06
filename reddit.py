import os
import torch
import argparse
import numpy as np
import scipy.sparse as sp
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn.models import LabelPropagation

from metrics import f1
from time import perf_counter
from utils import load_reddit_data, set_seed, EarlyStopping, modify_structure, to_embedding, get_model_and_precompute

def training_process(args, features, labels, adj, idx_train, idx_val, idx_test):
    model, processed_features, precompute_time, _ = get_model_and_precompute(
        args, adj, features, labels.max().item()+1)
    model = model.cuda()
    if args.model == "SIGN":
        features_list = [features] + [processed_features[i]
                                      for i in range(args.model_degree)]
        train_features = [x[idx_train].cuda() for x in features_list]
        val_features = [x[idx_val].cuda() for x in features_list]
        test_features = [x[idx_test].cuda() for x in features_list]
    else:
        train_features = processed_features[idx_train].cuda()
        val_features = processed_features[idx_val].cuda()
        test_features = processed_features[idx_test].cuda()
    labels = labels.cuda()

    best_val_f1, test_f1, train_time = train_eval(model, train_features, labels[idx_train], val_features, labels[idx_val], test_features, labels[idx_test], args)
    return best_val_f1, test_f1, train_time, precompute_time


def train_eval(model, train_features, train_labels, val_features, val_labels, test_features, test_labels, args):

    optimizer = optim.LBFGS(model.parameters(), lr=args.lr)
    early_stopping = EarlyStopping(5, use_loss=False)

    def closure():
        optimizer.zero_grad()
        output = model(train_features)
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        return loss_train
    
    t = perf_counter()
    for epoch in range(args.epochs):
        model.train()
        loss_train = optimizer.step(closure)
        with torch.no_grad():
            model.eval()
            output_val = model(val_features)
            preds_val = output_val.max(1)[1]
            val_f1, _ = f1(output_val, val_labels)
            output_test = model(test_features)
            preds_test = output_test.max(1)[1]
            test_f1, _ = f1(model(test_features), test_labels)
            early_stopping(val_f1, test_f1, model, (preds_val, preds_test))
            if early_stopping.early_stop:
                print(f'early stop at {epoch}')
                break
    
    train_time = perf_counter()-t
    return early_stopping.best_score, early_stopping.test_score, train_time

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='reddit', help='Dataset name.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--train_ori', action='store_true', default=False, help='whether to train on original graph')
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--alpha', type=float, default=0.05, help='SSGC precompute alpha')
parser.add_argument('--lr', type=float, default=1, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs to train.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--normalization', type=str, default='AugNormAdj',
                   choices=['NormLap', 'Lap', 'RWalkLap', 'FirstOrderGCN',
                            'AugNormAdj', 'NormAdj', 'RWalk', 'AugRWalk', 'NoNorm'],
                   help='Normalization method for the adjacency matrix.')
parser.add_argument('--model', type=str, default="SGC",
                    help='model to use.')
parser.add_argument('--degree', type=int, default=2,
                    help='degree of the approximation.')
parser.add_argument('--model_degree', type=int, default=2,
                    help='degree of the approximation.')
parser.add_argument('--lp_num_layers', type=int,
                        default=3, help='label propagation num layers')
parser.add_argument('--lp_alpha', type=float,
                        default=0.4, help='label propagation alpha')
parser.add_argument('--noise', type=str, default="add_delete", choices=['delete', 'add', 'add_delete', 'none'], 
                    help='the type of noise.')
parser.add_argument('--update', type=int, default=3, help='structure update times.')
parser.add_argument('--walk_len', type=int, default=5, help="length of random walk") #citeseer设置为8，cora设置为5
parser.add_argument('--high',type=float,default=0.8,help="threshold of adding edge")
parser.add_argument('--low',type=float,default=0.1,help="threshold of deleting edge")
parser.add_argument('--topk',type=int,default=4,help="select the most top k similar neighbor nodes")
parser.add_argument('--separate_1', type=int, default=3, help="degree threshold 1")
parser.add_argument('--separate_2', type=int, default=8, help="degree threshold 2")
parser.add_argument('--coefficient',type=float,default=0.6, help="coefficient in sampling")
parser.add_argument('--first_coe', type=float, default=0.5)
parser.add_argument('--second_coe', type=float, default=0.25)
parser.add_argument('--third_coe', type=float, default=0.5)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

set_seed(args.seed, args.cuda)

adj, adj_train, features, labels, p_labels, idx_train, idx_val, idx_test = load_reddit_data(args.noise)
print("Finished data loading.")

if args.train_ori:
    test_f1_list = []
    for _ in range(10):
        best_val_f1, test_f1, train_time, precompute_time = training_process(args, features, labels, adj, idx_train, idx_val, idx_test)
        test_f1_list.append(test_f1.item())
    print("#"*30)
    print("Original Graph Test\nMean Test F1: {:.4f}, Std: {:.4f}".format(np.mean(test_f1_list), np.std(test_f1_list)))

if os.path.exists(f"data/optimized_reddit_adj_{args.noise}_modified.npz"):
    adj = sp.load_npz(f"data/optimized_reddit_adj_{args.noise}_modified.npz")
    test_f1_list = []
    for _ in range(10):
        best_val_f1, test_f1, train_time, precompute_time = training_process(args, features, labels, adj, idx_train, idx_val, idx_test)
        test_f1_list.append(test_f1.item())
    print("Finally -- Optimized Augmented Graph Test\nMean Test F1: {:.4f}, Std: {:.4f}".format(np.mean(test_f1_list), np.std(test_f1_list)))

else:
    lp = LabelPropagation(args.lp_num_layers, args.lp_alpha)
    n = 0
    while n < args.update:
        final_features, smooth_labels = to_embedding(adj, features, args.degree, p_labels, lp, normalization="AugNormAdj", trans_type=args.model, alpha=args.alpha)
        adj = modify_structure(adj, features, smooth_labels, args)
        n += 1
        sp.save_npz(f"data/reddit_adj_{args.noise}_modified_{n}.npz", adj.tocsr())