import os
import torch
import argparse
import numpy as np
import scipy.sparse as sp
import torch.optim as optim
from time import perf_counter
import torch.nn.functional as F

from metrics import f1
from load_data import load_hetero
from method import modify_structure
from utils import set_seed, EarlyStopping, get_model, precompute, get_smooth_labels

def training_process(args, adj_tensor, processed_features, labels, nfeat, idx_train, idx_val, idx_test, device):
    model = get_model(args, nfeat, labels.max().item() + 1).to(device)

    processed_features = processed_features.to(device)
    adj_tensor = adj_tensor.to(device)
    labels = labels.to(device)

    best_val_f1, test_f1, train_time = train_eval(model, processed_features, labels, adj_tensor, args, idx_train, idx_val, idx_test)
    return best_val_f1, test_f1, train_time


def train_eval(model, features, labels, adj, args, idx_train, idx_val, idx_test):

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(30, use_loss=False)

    t = perf_counter()
    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            output = model(features, adj)
            val_maf1 = f1(output[idx_val], labels[idx_val])[1]
            test_mif1, test_maf1 = f1(output[idx_test], labels[idx_test])
            early_stopping(val_maf1, (test_mif1, test_maf1), model)
            if early_stopping.early_stop:
                break

    train_time = perf_counter() - t
    return early_stopping.best_score, early_stopping.test_score, train_time

if __name__ == "__main__":
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="acm", help="Dataset name.")
    parser.add_argument(
        "--device", type=int, default=0, help="GPU ID or CPU (-1)."
    )
    parser.add_argument(
        "--train_ori",
        action="store_true",
        default=False,
        help="whether to train on original graph",
    )
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--lr", type=float, default=0.009, help="Initial learning rate.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train.")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.001,
        help="Weight decay (L2 loss on parameters).",
    )
    parser.add_argument(
        "--random_sample",
        action="store_true",
        default=False,
        help="whether to do neighborhood sampling",
    )
    parser.add_argument("--model", type=str, default="GCN-hetero", help="model to use.")
    parser.add_argument('--lp_num_layers', type=int,
                        default=3, help='label propagation num layers')
    parser.add_argument('--lp_alpha', type=float,
                        default=0.4, help='label propagation alpha')
    parser.add_argument("--update", type=int, default=3, help="structure update times.")
    parser.add_argument(
        "--walk_len", type=int, default=5, help="length of random walk"
    ) 
    parser.add_argument("--high", type=float, default=0.8, help="threshold of adding edge")
    parser.add_argument("--low", type=float, default=0.1, help="threshold of deleting edge")
    parser.add_argument(
        "--topk", type=int, default=-1, help="select the most top k similar neighbor nodes"
    )
    parser.add_argument("--first_coe", type=float, default=0.5)
    parser.add_argument("--second_coe", type=float, default=0.25)
    parser.add_argument("--third_coe", type=float, default=0.5)
    parser.add_argument("--fourth_coe", type=float, default=1.)
    
    args = parser.parse_args()
    device = f"cuda:{args.device}" if args.device > -1 else "cpu"

    set_seed(args.seed, args.device!=-1)

    adj, features, labels, p_labels, idx_train, idx_val, idx_test = load_hetero(args.dataset)
    nfeat = features.shape[1]
    print("Finished data loading.")

    if args.train_ori:
        test_mif1_list = []
        test_maf1_list = []
        processed_features, _, adj_tensor = precompute(args, adj, features.numpy())

        for _ in range(10):
            best_val_maf1, test_f1, train_time = training_process(args, adj_tensor, processed_features, labels, nfeat, idx_train, idx_val, idx_test, device)
            test_mif1_list.append(test_f1[0].item())
            test_maf1_list.append(test_f1[1].item())
        print("#" * 30)
        print(
            "Original Graph Test\nMean Test Micro F1: {:.4f}, Std: {:.4f}\nMean Test Macro F1: {:.4f}, Std: {:.4f}".format(
                np.mean(test_mif1_list), np.std(test_mif1_list), np.mean(test_maf1_list), np.std(test_maf1_list)
            )
        )

    n = 0
    while n < args.update:
        smooth_labels = get_smooth_labels(adj, p_labels, args.lp_num_layers, args.lp_alpha)
        adj = modify_structure(adj, features, smooth_labels, args).tocsr()
        n += 1

    test_mif1_list = []
    test_maf1_list = []

    processed_features, _, adj_tensor = precompute(args, adj, features.numpy())
    for _ in range(10):
        best_val_maf1, test_f1, train_time = training_process(args, adj_tensor, processed_features, labels, nfeat, idx_train, idx_val, idx_test, device)
        test_mif1_list.append(test_f1[0].item())
        test_maf1_list.append(test_f1[1].item())
    print(
        "Finally -- Optimized Augmented Graph Test\nMean Test Micro F1: {:.4f}, Std: {:.4f}\nMean Test Macro F1: {:.4f}, Std: {:.4f}".format(
            np.mean(test_mif1_list), np.std(test_mif1_list), np.mean(test_maf1_list), np.std(test_maf1_list)
        )
    )
