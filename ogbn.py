import os 
import torch
import argparse
import numpy as np
import scipy.sparse as sp
from time import perf_counter
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from load_data import load_ogb_data
from method import modify_structure
from utils import (
    set_seed,
    EarlyStopping,
    get_smooth_labels,
    get_model,
    precompute
)

def training_process(args, evaluator, adj_tensor, features, processed_features, labels, nfeat, idx_train, idx_val, idx_test, device):
    model = get_model(args, nfeat, labels.max().item() + 1).to(device)

    if args.model == "SAGE":
        best_val_acc, test_acc, train_time, preds = train_eval_SAGE(model, evaluator, processed_features, labels, adj_tensor, args, idx_train, idx_val, idx_test)
    else:
        if args.model == "SIGN":
            features_list = [features] + [
                processed_features[i] for i in range(args.model_degree)
            ]
            train_features = [x[idx_train] for x in features_list]
            val_features = [x[idx_val] for x in features_list]
            test_features = [x[idx_test] for x in features_list]
        else:
            train_features = processed_features[idx_train]
            val_features = processed_features[idx_val]
            test_features = processed_features[idx_test]

        best_val_acc, test_acc, train_time, preds = train_eval(
            model,
            evaluator,
            train_features,
            labels.squeeze(1)[idx_train],
            val_features,
            labels[idx_val],
            test_features,
            labels[idx_test],
            args,
            device
        )

    return best_val_acc, test_acc, train_time, preds


@torch.no_grad()
def test(model, features, y_true, evaluator, device, batch_size=-1):
    model.eval()
    y_preds = []
    if batch_size > 0:
        loader = DataLoader(range(y_true.size(0)), batch_size=batch_size)

        for perm in loader:
            y_pred = (
                F.log_softmax(model([x[perm].to(device) for x in features]), dim=-1).argmax(
                    dim=-1, keepdim=True
                )
                if isinstance(features, list)
                else F.log_softmax(model(features[perm].to(device)), dim=-1).argmax(
                    dim=-1, keepdim=True
                )
            )
            y_preds.append(y_pred.cpu())
        y_pred = torch.cat(y_preds, dim=0)
    else:
        y_pred = F.log_softmax(model(features.to(device)), dim=-1).argmax(
            dim=-1, keepdim=True
        )
        y_pred = y_pred.cpu()

    return (
        evaluator.eval(
            {
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )["acc"],
        y_pred,
    )

def train_eval_SAGE(model, evaluator, features, labels, adj, args, idx_train, idx_val, idx_test):
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    early_stopping = EarlyStopping(args.max_patience, use_loss=False)
    
    train_batch_size = 1024
    eval_batch_size = 4096
    
    graph = Data(features, adj, y=labels, n_id=torch.arange(features.size(0)))
    graph = graph.to(device, 'y')
    train_loader = NeighborLoader(graph, input_nodes=idx_train, num_neighbors=[15, 10, 5], \
                                  batch_size=train_batch_size, shuffle=True, num_workers=0)
    eval_loader = NeighborLoader(graph, input_nodes=None, num_neighbors=[-1], batch_size=eval_batch_size, num_workers=0)
    
    t = perf_counter()

    for epoch in range(1, 21):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x.to(device), batch.edge_index.to(device))[:batch.batch_size]
            y = batch.y[:batch.batch_size].squeeze()
            loss = F.cross_entropy(out, y)
            total_loss += float(loss)
            loss.backward()
            optimizer.step()
        total_loss /= len(train_loader)
        print(f"train loss: {total_loss:.4f}")
        
        if epoch > 0:
            with torch.no_grad():
                model.eval()
                out = model.inference(graph.x, eval_loader, device)
                y_true = graph.y.cpu()
                y_pred = out.argmax(dim=-1, keepdim=True)
                acc_val = evaluator.eval({"y_true": y_true[idx_val], "y_pred": y_pred[idx_val]})["acc"]
                acc_test = evaluator.eval({"y_true": y_true[idx_test], "y_pred": y_pred[idx_test]})["acc"]
                print(f"val accuracy: {acc_val:.4f}, test accuracy: {acc_test:.4f}")
                early_stopping(acc_val, acc_test, model, y_pred)
                if early_stopping.early_stop:
                    break

    train_time = perf_counter() - t
    return early_stopping.best_score, early_stopping.test_score, train_time, early_stopping.preds

def train_eval(
    model,
    evaluator,
    train_features,
    train_labels,
    val_features,
    val_labels,
    test_features,
    test_labels,
    args,
    device
):
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    early_stopping = EarlyStopping(args.max_patience, use_loss=False)

    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size

    def closure():
        train_size = (
            train_features[0].size(0)
            if isinstance(train_features, list)
            else train_features.size(0)
        )
        if train_batch_size > 0:
            loader = DataLoader(
                range(train_size), batch_size=train_batch_size, shuffle=True
            )
            for perm in loader:
                optimizer.zero_grad()
                output = (
                    F.log_softmax(model([x[perm].to(device) for x in train_features]), dim=-1)
                    if isinstance(train_features, list)
                    else F.log_softmax(model(train_features[perm].to(device)), dim=-1)
                )
                loss_train = F.nll_loss(output, train_labels[perm].to(device))
                loss_train.backward()
        else:
            optimizer.zero_grad()
            output = (
                F.log_softmax(model([x.to(device) for x in train_features]), dim=-1)
                if isinstance(train_features, list)
                else F.log_softmax(model(train_features.to(device)), dim=-1)
            )
            loss_train = F.nll_loss(output, train_labels.to(device))
            loss_train.backward()

    t = perf_counter()
    for epoch in range(1, args.epochs+1):
        model.train()
        optimizer.step(closure)

        if epoch % args.eval_every == 0 and epoch >= args.eval_start:
            val_acc, val_preds = test(
                model, val_features, val_labels, evaluator, device, eval_batch_size
            )
            test_acc, test_preds = test(
                model, test_features, test_labels, evaluator, device, eval_batch_size
            )
            early_stopping(val_acc, test_acc, model, (val_preds, test_preds))
            if early_stopping.early_stop:
                break

    return early_stopping.best_score, early_stopping.test_score, perf_counter() - t, early_stopping.preds

if __name__ == "__main__":
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="GPU ID or CPU (-1).")
    parser.add_argument("--train_ori", action="store_true", default=False, help="whether to train on original graph")
    parser.add_argument('--pool_num', type=int, default=5, help="pool nums")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--ff_layer", type=int, default=2, help="number of feed-forward layers")
    parser.add_argument("--hidden_channels", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--input_drop", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--alpha", type=float, default=0.05, help="SSGC precompute alpha.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train.")
    parser.add_argument("--eval_every", type=int, default=10, help="eval test evert x epochs.")
    parser.add_argument("--eval_start", type=int, default=10, help="eval start at x epoch.")
    parser.add_argument("--max_patience", type=int, default=5, help="patience for early stop.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay (L2 loss on parameters).")
    parser.add_argument("--train_batch_size", type=int, default=100000, help="train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=200000, help="eval batch size")
    parser.add_argument("--normalization", type=str, default="AugNormAdj", choices=["AugNormAdj", "RWalk"],
        help="Normalization method for the adjacency matrix.")
    parser.add_argument("--model", type=str, default="SGC", choices=["SGC", "SIGN", "SAGE"], help="model to use.")
    parser.add_argument("--dataset", type=str, default="ogbn_products", help="ogbn dataset name")
    parser.add_argument("--model_degree", type=int, default=5, help="degree of the approximation.")
    parser.add_argument("--lp_num_layers", type=int, default=3, help="label propagation num layers")
    parser.add_argument("--lp_alpha", type=float, default=0.4, help="label propagation alpha")
    parser.add_argument("--noise", type=str, default="none", choices=["delete", "add", "add_delete", "none"], help="the type of noise.")
    parser.add_argument("--update", type=int, default=2, help="structure update times.")
    parser.add_argument("--walk_len", type=int, default=5, help="length of random walk")
    parser.add_argument("--high", type=float, default=0.8, help="threshold of adding edge")
    parser.add_argument("--low", type=float, default=0.1, help="threshold of deleting edge")
    parser.add_argument("--topk", type=int, default=-1, help="select the most top k similar neighbor nodes")
    parser.add_argument('--random_sample', action='store_true', default=False, help="whether to do neighborhood sampling")
    parser.add_argument('--separate_1', type=int, default=3, help="degree threshold 1")
    parser.add_argument('--first_coe', type=float, default=0.5)
    parser.add_argument('--second_coe', type=float, default=0.25)
    parser.add_argument('--third_coe', type=float, default=0.5)
    parser.add_argument('--fourth_coe', type=float, default=1.)

    args = parser.parse_args()
    device = f"cuda:{args.device}" if args.device > -1 else "cpu"

    set_seed(args.seed, args.device!=-1)
    start_t = perf_counter()
    adj, features, labels, p_labels, idx_train, idx_val, idx_test, evaluator = load_ogb_data(args.dataset, noise=args.noise)
    p_labels = p_labels.squeeze(1)
    print("Finished data loading.")

    if args.train_ori:
        test_acc_list = []
        processed_features, _, adj_tensor = precompute(args, adj, features)
        for _ in range(10):
            best_val_acc, test_acc, train_time, _ = training_process(
                args, evaluator, adj_tensor, features, processed_features, labels, features.shape[1], idx_train, idx_val, idx_test, device
            )
            print(f"TEST ACC: {test_acc:.4f}")
            test_acc_list.append(test_acc)
        print("#" * 30)
        print(
            "Original Graph Test\nMean Test Acc: {:.4f}, Std: {:.4f}".format(
                np.mean(test_acc_list), np.std(test_acc_list)
            )
        )

    n = 0
    while n < args.update:
        smooth_labels = get_smooth_labels(adj, p_labels, args.lp_num_layers, args.lp_alpha)
        adj = modify_structure(adj, features, smooth_labels, args).tocsr()
        n += 1
    end_t = perf_counter()
    print(f"Optimization process takes {(end_t - start_t)/60} mins")
    
    test_acc_list = []
    processed_features, _, adj_tensor = precompute(args, adj, features)
    
    for _ in range(10):
        best_val_acc, test_acc, train_time, _ = training_process(
                args, evaluator, adj_tensor, features, processed_features, labels, features.shape[1], idx_train, idx_val, idx_test, device
            )
        test_acc_list.append(test_acc)

    print("Finally -- Optimized Augmented Graph Test\nMean Test Acc: {:.4f}, Std: {:.4f}" \
        .format(np.mean(test_acc_list), np.std(test_acc_list)))
