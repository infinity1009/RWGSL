import os
import torch
import numpy as np
import scipy.sparse as sp
import torch.optim as optim
from time import perf_counter
import torch.nn.functional as F

from args import get_args
from metrics import accuracy
from sample import isolate_node
from method import modify_structure
from load_data import load_citation_data
from utils import set_seed, get_smooth_features, get_smooth_labels, EarlyStopping, get_model, \
    visual_feature_similarity, precompute


def training_process(args, adj_tensor, processed_features, labels, nfeat, idx_train, idx_val, idx_test, device):
    model = get_model(args, nfeat, labels.max().item() + 1).to(device)
    
    if args.model == "SIGN":
        features_list = [features] + [processed_features[i]
                                      for i in range(args.model_degree)]
        processed_features = [x.to(device) for x in features_list]
    elif args.model == "SGC":
        processed_features = processed_features.to(device)
    elif args.model in ["GCN", "SAGE", "GAT"]:
        processed_features = processed_features.to(device)
        adj_tensor = adj_tensor.to(device)

    labels = labels.to(device)
    best_val_acc, test_acc, train_time, preds = train_eval(model, processed_features, labels, adj_tensor, args, idx_train, idx_val, idx_test)
    
    return best_val_acc, test_acc, train_time, preds


def train_eval(model, features, labels, adj, args, idx_train, idx_val, idx_test):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(50, use_loss=False)
    t = perf_counter()

    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        if args.model in ["GCN", "SAGE", "GAT"]:
            output = model(features, adj)
        else:
            output = model(features)
        loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
        with torch.no_grad():
            model.eval()
            if args.model in ["GCN", "SAGE", "GAT"]:
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
    return early_stopping.best_score, early_stopping.test_score, train_time, early_stopping.preds

if __name__ == "__main__":
    # Arguments
    args = get_args()
    args.num_samples = [int(layer_size) for layer_size in args.num_samples.split(',')]

    # setting random seeds
    set_seed(args.seed, args.device!=-1)
    device = f"cuda:{args.device}" if args.device > -1 else "cpu"
    start_t = perf_counter()
    adj, features, labels, p_labels, idx_train, idx_val, idx_test = load_citation_data(args.dataset, args.drop_rate, args.add_rate, args.mask_feat_rate, args.label_per_class)
    nfeat = features.shape[1]
    print(f"Finish loading {args.dataset}.")
    if args.train_ori:
        test_acc_list = []
        processed_features, _, adj_tensor = precompute(args, adj, features.numpy())
        for _ in range(10):
            best_val_acc, test_acc, train_time, _ = training_process(args, adj_tensor, processed_features, labels, nfeat, idx_train, idx_val, idx_test, device)
            test_acc_list.append(test_acc.item())
        print("#" * 30)
        print("All 10 results: ", test_acc_list)
        print("Original Graph Test\nTotal Time: {:.4f}s, Mean Test Acc: {:.4f}, Std: {:.4f}".format(
            train_time, np.mean(test_acc_list), np.std(test_acc_list)))
    
    if os.path.exists(f"data/optimized/{args.dataset}_adj_none_modified.npz"):
        adj = sp.load_npz(f"data/optimized/{args.dataset}_adj_none_modified.npz")

        test_acc_list = []
        processed_features, _, adj_tensor = precompute(args, adj, features.numpy())

        for _ in range(10):
            best_val_acc, test_acc, train_time, _ = training_process(args, adj_tensor, processed_features, labels, nfeat, idx_train, idx_val, idx_test, device)
            test_acc_list.append(test_acc.item())
        print("Finally -- Optimized Augmented Graph Test\nMean Test Acc: {:.4f}, Std: {:.4f}".format(
            np.mean(test_acc_list), np.std(test_acc_list)))
   
    else:
        start_o = perf_counter()
        n = 0
        while n < args.update: 
            start_i = perf_counter()
            smooth_labels = get_smooth_labels(adj, p_labels, args.lp_num_layers, args.lp_alpha)
            adj = modify_structure(adj, features, smooth_labels, args).tocsr()
            end_i = perf_counter()
            if args.verbose:
                smooth_features = get_smooth_features(adj, features, args.degree, normalization="AugNormAdj", trans_type="SGC")
                distribution, total_edge = visual_feature_similarity(adj, smooth_features)
                print("Embedding相似性分布", distribution)
                print("此时，总边数为:", total_edge)

                time_list = []
                test_acc_list = []
                processed_features, _, adj_tensor = precompute(args, adj, features.numpy())
                
                for _ in range(10):
                    start_t = perf_counter()
                    best_val_acc, test_acc, train_time, _ = training_process(args, adj_tensor, processed_features, labels, nfeat, idx_train, idx_val, idx_test, device)
                    test_acc_list.append(test_acc.item())
                    time_list.append(perf_counter()-start_t)
                
                print(f"Mean training time: {np.mean(time_list):.4f}, Optimization takes: {end_i-start_i:.4f} sec")
                print("Iteration: {} -- Optimized Augmented Graph Test\nMean Test Acc: {:.4f}, Std: {:.4f}".format(
                    n, np.mean(test_acc_list), np.std(test_acc_list)))
            
            n += 1
        
        print(f"Optimization process takes: {perf_counter() - start_o:.4f} sec")

        if not args.verbose:
            time_list = []
            test_acc_list = []
            processed_features, _, adj_tensor = precompute(args, adj, features.numpy())
            
            for _ in range(10):
                start_t = perf_counter()
                best_val_acc, test_acc, train_time, _ = training_process(args, adj_tensor, processed_features, labels, nfeat, idx_train, idx_val, idx_test, device)
                test_acc_list.append(test_acc.item())
                time_list.append(perf_counter()-start_t)
            
            print("Final: {} -- Optimized Augmented Graph Test\nMean Test Acc: {:.4f}, Std: {:.4f}".format(
                n, np.mean(test_acc_list), np.std(test_acc_list)))
        
        if args.verbose:
            adj1 = torch.tensor(adj.todense())
            print("孤立节点数为:", isolate_node(adj1))
            smooth_features = get_smooth_features(adj, features, args.degree, normalization="AugNormAdj", trans_type="SGC")
            distribution, total_edge = visual_feature_similarity(adj, smooth_features)
            print("Embedding相似性分布", distribution)
            print("此时，总边数为:", total_edge)