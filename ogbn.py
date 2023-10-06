import os
import torch
import argparse
import numpy as np
import scipy.sparse as sp
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn.models import LabelPropagation

from time import perf_counter
from torch.utils.data import DataLoader
from utils import load_ogb_data, set_seed, EarlyStopping, modify_structure, to_embedding, get_model_and_precompute


def training_process(args, evaluator, features, labels, adj, idx_train, idx_val, idx_test):
    model, processed_features, precompute_time, _ = get_model_and_precompute(
        args, adj, features, labels.max().item()+1)
    model = model.cuda()
    if args.model == "SIGN":
        features_list = [features] + [processed_features[i]
                                      for i in range(args.model_degree)]
        train_features = [x[idx_train]for x in features_list]
        val_features = [x[idx_val]for x in features_list]
        test_features = [x[idx_test] for x in features_list]
    else:
        train_features = processed_features[idx_train]
        val_features = processed_features[idx_val]
        test_features = processed_features[idx_test]

    best_val_acc, test_acc, train_time = train_eval(model, evaluator, train_features, labels.squeeze(
        1)[idx_train], val_features, labels[idx_val], test_features, labels[idx_test], args)
    return best_val_acc, test_acc, train_time, precompute_time


@torch.no_grad()
def test(model, features, y_true, evaluator, batch_size = -1):
    model.eval()
    y_preds = []
    if batch_size > 0:
        loader = DataLoader(range(y_true.size(0)), batch_size=batch_size)
        
        for perm in loader:
            y_pred = F.log_softmax(model([x[perm].cuda() for x in features]), dim=-1).argmax(dim=-1, keepdim=True) if isinstance(
                features, list) else F.log_softmax(model(features[perm].cuda()), dim=-1).argmax(dim=-1, keepdim=True)
            y_preds.append(y_pred.cpu())
        y_pred = torch.cat(y_preds, dim=0)
    
    else:
        y_pred = F.log_softmax(model(features).cuda(), dim=-1).argmax(dim=-1, keepdim=True)
        y_pred = y_pred.cpu()

    return evaluator.eval({
        'y_true': y_true,
        'y_pred': y_pred,
    })['acc'], y_pred


def train_eval(model, evaluator, train_features, train_labels, val_features, val_labels, test_features, test_labels, args):

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(30, use_loss=False)

    train_batch_size = 200000
    batch_size = 200000 if args.dataset == "ogbn_products" else -1

    def closure():
        train_size = train_features[0].size(0) if isinstance(train_features, list) else train_features.size(0)
        loader = DataLoader(range(train_size), batch_size=train_batch_size, shuffle=True)

        total_loss = 0.
        for perm in loader:
            optimizer.zero_grad()
            output = F.log_softmax(model([x[perm].cuda() for x in train_features]), dim=-1) if isinstance(
                train_features, list) else F.log_softmax(model(train_features[perm].cuda()), dim=-1)
            loss_train = F.nll_loss(output, train_labels[perm].cuda())
            total_loss += loss_train * perm.size(0)
            loss_train.backward()

        return total_loss / train_size

    t = perf_counter()
    for epoch in range(args.epochs):
        model.train()
        loss_train = optimizer.step(closure)

        val_acc, val_preds = test(model, val_features, val_labels, evaluator, batch_size)
        test_acc, test_preds = test(model, test_features, test_labels, evaluator, batch_size)
        early_stopping(val_acc, test_acc, model, (val_preds, test_preds))
        if early_stopping.early_stop:
            print(f'early stop at {epoch}')
            break

    train_time = perf_counter() - t
    return early_stopping.best_score, early_stopping.test_score, train_time


# Args
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--inductive', action='store_true', default=False,
                    help='inductive training.')
parser.add_argument('--train_ori', action='store_true', default=False, help='whether to train on original graph')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--alpha', type=float, default=0.05, help='SSGC precompute alpha')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--normalization', type=str, default='AugNormAdj',
                    choices=['NormLap', 'Lap', 'RWalkLap', 'FirstOrderGCN',
                             'AugNormAdj', 'NormAdj', 'RWalk', 'AugRWalk', 'NoNorm'],
                    help='Normalization method for the adjacency matrix.')
parser.add_argument('--model', type=str, default="SGC",
                    help='model to use.')
parser.add_argument('--dataset', type=str, default="ogbn_products", help='ogbn dataset name')
parser.add_argument('--degree', type=int, default=3,
                    help='degree of the approximation.')
parser.add_argument('--model_degree', type=int, default=2,
                    help='degree of the approximation.')
parser.add_argument('--lp_num_layers', type=int,
                        default=3, help='label propagation num layers')
parser.add_argument('--lp_alpha', type=float,
                        default=0.4, help='label propagation alpha')
parser.add_argument('--noise', type=str, default="add_delete", choices=['delete', 'add', 'add_delete', 'none'],
                    help='the type of noise.')
parser.add_argument('--update', type=int, default=2,
                    help='structure update times.')
parser.add_argument('--walk_len', type=int, default=5,
                    help="length of random walk")
parser.add_argument('--high', type=float, default=0.8,
                    help="threshold of adding edge")
parser.add_argument('--low', type=float, default=0.1,
                    help="threshold of deleting edge")
parser.add_argument('--topk', type=int, default=4,
                    help="select the most top k similar neighbor nodes")
parser.add_argument('--sample_number', type=int, default=4,
                    help="high threshold of sampled nodes")
parser.add_argument('--coefficient', type=float,
                    default=0.2, help="coefficient in sampling")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

set_seed(args.seed, args.cuda)
adj, features, labels, p_labels, idx_train, idx_val, idx_test, dataset, evaluator = load_ogb_data(
    args.noise, args.dataset)
print("Finished data loading.")

if args.train_ori:
    test_acc_list = []
    for _ in range(10):
        best_val_acc, test_acc, train_time, precompute_time = training_process(
            args, evaluator, features, labels, adj, idx_train, idx_val, idx_test)
        test_acc_list.append(test_acc)
    print("#"*30)
    print("Original Graph Test\nMean Test Acc: {:.4f}, Std: {:.4f}".format(np.mean(test_acc_list), np.std(test_acc_list)))

if os.path.exists(f"data/optimized_{args.dataset}_adj_{args.noise}_modified.npz"):
    adj = sp.load_npz(f"data/optimized_{args.dataset}_adj_{args.noise}_modified.npz")
    test_acc_list = []
    for _ in range(10):
        start_time = perf_counter()
        best_val_acc, test_acc, train_time, precompute_time = training_process(
            args, evaluator, features, labels, adj, idx_train, idx_val, idx_test)
        test_acc_list.append(test_acc)
        print(f"running time: {perf_counter()-start_time}")
    print("Finally -- Optimized Augmented Graph Test\nMean Test Acc: {:.4f}, Std: {:.4f}".format(
        np.mean(test_acc_list), np.std(test_acc_list)))

else:
    lp = LabelPropagation(args.lp_num_layers, args.lp_alpha)
    n = 0
    while n < args.update:
        final_features, smooth_labels = to_embedding(
            adj, features, args.degree, p_labels, lp, normalization="AugNormAdj", trans_type=args.model, alpha=args.alpha)
        adj = modify_structure(adj, features, smooth_labels, args)
        n += 1
        sp.save_npz(
            f"data/{args.dataset}_adj_{args.noise}_modified_{n}.npz", adj.tocsr())
