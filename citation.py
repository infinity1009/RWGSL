import os
import torch
import numpy as np
import scipy.sparse as sp
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn.models import LabelPropagation

import sample
from args import get_args
from metrics import accuracy
from time import perf_counter
from utils import set_seed, load_data, modify_structure, to_embedding, EarlyStopping, get_model_and_precompute, \
    visual_feature_similarity


def training_process(args, features, labels, adj):
    model, processed_features, precompute_time, adj_tensor = get_model_and_precompute(
        args, adj, features, labels.max().item() + 1)
    model = model.cuda()
    if args.model == "SIGN":
        features_list = [features] + [processed_features[i]
                                      for i in range(args.model_degree)]
        processed_features = [x.cuda() for x in features_list]
    elif args.model == "SGC":
        processed_features = processed_features.cuda()
    elif args.model == "GCN":
        processed_features = processed_features.cuda()
        adj_tensor = adj_tensor.cuda()

    labels = labels.cuda()
    best_val_acc, test_acc, preds, train_time = train_eval(model, processed_features, labels, adj_tensor, args)
    return best_val_acc, test_acc, preds, train_time, precompute_time


def train_eval(model, features, labels, adj, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(20, use_loss=False)
    t = perf_counter()

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        if args.model == "GCN":
            output = model(features, adj)
        else:
            output = model(features)
        loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
        with torch.no_grad():
            model.eval()
            if args.model == "GCN":
                output = model(features, adj)
            else:
                output = model(features)
            preds = output.max(1)[1].type_as(labels)
            acc_val = accuracy(output[idx_val], labels[idx_val])
            acc_test = accuracy(output[idx_test], labels[idx_test])
            early_stopping(acc_val, acc_test, model, preds)
            if early_stopping.early_stop:
                break

    train_time = perf_counter() - t
    return early_stopping.best_score, early_stopping.test_score, early_stopping.preds, train_time


# Arguments
args = get_args()

# setting random seeds
set_seed(args.seed, args.cuda)

adj, features, labels, p_labels, idx_train, idx_val, idx_test = load_data(args.dataset, args.noise, args.normalization)
print(f"Finish loading {args.dataset}.")

if args.train_ori:
    test_acc_list = []
    for _ in range(10):
        best_val_acc, test_acc, _, train_time, precompute_time = training_process(args, features, labels, adj)
        test_acc_list.append(test_acc.item())
    print("#" * 30)
    print("Original Graph Test\nTotal Time: {:.4f}s, Mean Test Acc: {:.4f}, Std: {:.4f}".format(
        train_time + precompute_time, np.mean(test_acc_list), np.std(test_acc_list)))

if os.path.exists(f"data/optimized_{args.dataset}_adj_{args.noise}_modified.npz"):
    adj = sp.load_npz(f"data/optimized_{args.dataset}_adj_{args.noise}_modified.npz")

    test_acc_list = []
    for _ in range(10):
        best_val_acc, test_acc, _, train_time, precompute_time = training_process(args, features, labels, adj)
        test_acc_list.append(test_acc.item())
    print("Finally -- Optimized Augmented Graph Test\nMean Test Acc: {:.4f}, Std: {:.4f}".format(
        np.mean(test_acc_list), np.std(test_acc_list)))

else:
    lp = LabelPropagation(args.lp_num_layers, args.lp_alpha)
    n = 0
    while n < args.update: 
        final_features, smooth_labels = to_embedding(adj, features, args.degree, p_labels, lp, normalization="AugNormAdj", trans_type="SGC")
        if args.verbose:
            L, total_edge = visual_feature_similarity(adj, final_features)
            print("Embedding相似性分布", L)
            print("此时，总边数为:", total_edge)
        adj = modify_structure(adj, features, smooth_labels, args)
        n += 1
        sp.save_npz(f"data/{args.dataset}_adj_{args.noise}_modified_{n}.npz", adj.tocsr())

    if args.verbose:
        adj1 = torch.tensor(adj.todense())
        print("孤立节点数为:", sample.isolate_node(adj1))

        L, total_edge = visual_feature_similarity(adj, final_features)
        print("Embedding相似性分布", L)
        print("此时，总边数为:", total_edge)