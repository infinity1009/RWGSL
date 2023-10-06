from sample import calculate_similarity
import numpy as np
import scipy.sparse as sp
import torch
import os
import sys
import pickle as pkl
import networkx as nx
from normalization import fetch_normalization, row_normalize
from time import perf_counter
from sample import random_sample, modify
import numpy.random as random
from models import SGC, MLP, GCN
from torch_geometric.utils import from_scipy_sparse_matrix, remove_self_loops
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.col, sparse_mx.row)).astype(np.int64))  # 这里也调换了
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def add_super_node(adj): 
    adj = adj.todense() 
    n = adj.shape[0]
    D = adj.sum(axis=1)
    ind = np.argmax(D)
    for i in range(n):
        adj[ind, i] = 1
        adj[i, ind] = 1
    return sp.csr_matrix(adj)


def load_citation(dataset_str="cora", normalization="AugNormAdj", cuda=True):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))  # csr形式
    adj = adj + adj.T.multiply(adj.T > adj) - \
          adj.multiply(adj.T > adj)  # 变成对称的邻接矩阵
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()  # 转换成了tensor的稀疏表示
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def sgc_precompute(features, adj, degree, model="SGC", alpha=0.05):
    if model == "SSGC":
        t = perf_counter()
        emb = alpha * features
        for i in range(degree):
            features = torch.spmm(adj, features)
            emb = emb + (1 - alpha) * features / degree
        precompute_time = perf_counter() - t
        return emb, precompute_time

    else:
        t = perf_counter()
        for _ in range(degree):
            features = torch.mm(adj, features)
        precompute_time = perf_counter() - t
        return features, precompute_time


def sign_precompute(features, adj, num_layers):
    t = perf_counter()
    processed_list = []
    for _ in range(num_layers):
        features = torch.mm(adj, features)
        processed_list.append(features)
    precompute_time = perf_counter() - t
    return processed_list, precompute_time


def set_seed(seed, cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir + "reddit_adj.npz")
    data = np.load(dataset_dir + "reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], \
           data['test_index']


def load_reddit_data(noise_type, data_path="data/"):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ(
        data_path)
    labels = np.zeros(adj.shape[0])
    labels[train_index] = y_train
    labels[val_index] = y_val
    labels[test_index] = y_test
    adj = adj + adj.T  # csr-format
    features = torch.FloatTensor(np.array(features))

    train_adj = adj[train_index, :][:, train_index]

    features = (features - features.mean(dim=0)) / features.std(dim=0)

    labels = torch.LongTensor(labels)

    p_labels = torch.load(f'./data/reddit_{noise_type}_p_labels.pt')

    return adj, train_adj, features, labels, p_labels, train_index, val_index, test_index


def load_ogb_data(noise_type, name="ogbn-products", root="data"):
    p_labels = torch.load(f'./data/{name}_{noise_type}_p_labels.pt')

    name_map = {"ogbn_products": "ogbn-products", "ogbn_arxiv": "ogbn-arxiv"}
    name = name_map[name]
    dataset = PygNodePropPredDataset(name, root)
    splits = dataset.get_idx_split()
    train_index, val_index, test_index = splits["train"], splits["valid"], splits["test"]

    data = dataset[0]
    num_nodes = data.x.shape[0]
    row, col = remove_self_loops(data.edge_index)[0]
    adj = sp.coo_matrix((torch.ones_like(row), (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    adj = adj + adj.T

    evaluator = Evaluator(name)

    return adj, data.x, data.y, p_labels, train_index, val_index, test_index, dataset, evaluator


def to_embedding(adj, features, degree, labels, lp, normalization="AugNormAdj", trans_type="SGC", alpha=0.05):
    edge_index = from_scipy_sparse_matrix(adj)[0]
    smooth_labels = lp(labels, edge_index)
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).float()
    if trans_type == "SSGC":
        embedding = sgc_precompute(features, adj_tensor, degree, trans_type, alpha)[0]
    else:
        embedding = sgc_precompute(features, adj_tensor, degree)[0]
    return embedding, smooth_labels


def load_data(dataset_str="cora", noise="none", normalization="AugNormAdj"):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)) 
    adj = adj + adj.T.multiply(adj.T > adj) - \
          adj.multiply(adj.T > adj) 
    features = torch.FloatTensor(np.array(features.todense())).float()

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    adjusted_adj, features = preprocess_citation(adj, features, normalization)

    features = torch.FloatTensor(features).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sp.coo_matrix(adj)
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    p_labels = torch.load(f'data/{dataset_str}_{noise}_p_labels.pt')

    return adj, features, labels, p_labels, idx_train, idx_val, idx_test


def modify_structure(adj, raw_features, smooth_labels, args):
    adj = adj.tolil()
    sample_dict, neighbor_dict = random_sample(adj, s1=args.separate_1, s2=args.separate_2,
                                               m=args.coefficient) 
    adj = modify(adj, raw_features, smooth_labels,
                 sample_dict, neighbor_dict, topk=args.topk, low_threshold=args.low, high_threshold=args.high,
                 walk_len=args.walk_len, fst=args.first_coe, snd=args.second_coe, trd=args.third_coe)
    return adj


def to_torch_sparse_tensor(adj):
    adj = adj.cpu()
    adj = adj.numpy()
    adj = sp.coo_matrix(adj)
    indices = torch.from_numpy(
        np.vstack((adj.row, adj.col)).astype(np.int64))  
    values = torch.from_numpy(adj.data)
    shape = torch.Size(adj.shape)
    adj = torch.sparse.FloatTensor(indices, values, shape)
    adj = adj
    return adj


def visual_feature_similarity(adj, features):
    L = [0] * 10
    total_edge = 0
    adj = adj.todense()
    n = adj.shape[0]
    # O(n^2)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                total_edge += 1
                x = calculate_similarity(features, i, j)
                ind = int(x * 10)
                if ind == 10:
                    L[-1] += 1
                else:
                    L[ind] += 1
    return L, total_edge


def eval_simi(dict_add, k1, k2, k3, k4, final_features):
    m1, m2, m3, m4, m5 = 0, 0, 0, 0, 0
    n1, n2, n3, n4, n5 = 0, 0, 0, 0, 0
    for keys in dict_add:
        i, j = keys[0], keys[1]
        v = dict_add[keys][0]
        if v < k1:
            m1 += 1
        elif k1 < v < k2:
            m2 += 1
        elif k2 < v < k3:
            m3 += 1
        elif k3 < v < k4:
            m4 += 1
        else:
            m5 += 1
        s = calculate_similarity(final_features, i, j)
        if s < k1:
            n1 += 1
        elif k1 < s < k2:
            n2 += 1
        elif k2 < s < k3:
            n3 += 1
        elif k3 < s < k4:
            n4 += 1
        else:
            n5 += 1
        dict_add[(i, j)].append(s)
    L_add_before = [m1, m2, m3, m4, m5]
    L_add_after = [n1, n2, n3, n4, n5]
    return L_add_before, L_add_after


def get_model_and_precompute(args, adj, features, nclass):
    if args.model in ["SGC", "SSGC"]:
        model = SGC(features.shape[1], nclass)
        adj_normalizer = fetch_normalization(args.normalization)
        adj_1 = adj_normalizer(adj)
        adj_tensor_1 = sparse_mx_to_torch_sparse_tensor(adj_1).float()
        if not args.dataset.startswith("ogbn"):
            adj_tensor_1 = adj_tensor_1.cuda()
            features = features.cuda()
        processed_features, precompute_time = sgc_precompute(features, adj_tensor_1, args.model_degree, args.model,
                                                             args.alpha)
    elif args.model == "SIGN":
        model = MLP(features.shape[1], args.hidden_channels, nclass, args.model_degree, args.dropout)
        adj_normalizer = fetch_normalization(args.normalization)
        adj_1 = adj_normalizer(adj)
        adj_tensor_1 = sparse_mx_to_torch_sparse_tensor(adj_1).float()
        if not args.dataset.startswith("ogbn"):
            adj_tensor_1 = adj_tensor_1.cuda()
            features = features.cuda()
        processed_features, precompute_time = sign_precompute(features, adj_tensor_1, args.model_degree)

    elif args.model == "GCN":
        model = GCN(features.shape[1], args.hidden_channels, nclass, args.dropout)
        adj_normalizer = fetch_normalization(args.normalization)
        adj_1 = adj_normalizer(adj)
        adj_tensor_1 = sparse_mx_to_torch_sparse_tensor(adj_1).float().cuda()
        features = features.cuda()
        processed_features, precompute_time = features, 0

    return model, processed_features, precompute_time, adj_tensor_1


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience, save_model=False, verbose=False, delta=0., save_path='./', use_loss=True):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.save_model = save_model
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_score_max = 0.
        self.test_score = 0.
        self.delta = delta
        self.save_path = save_path + 'checkpoint.pt'
        self.use_loss = use_loss
        self.preds = None

    def __call__(self, val, test, model, preds):

        if self.use_loss:
            score = -val
        else:
            score = val

        if self.best_score is None:
            self.best_score = score
            self.preds = preds
            if self.save_model:
                self.save_checkpoint(val, model)
            self.test_score = test
        elif score < self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.preds = preds
            if self.save_model:
                self.save_checkpoint(val, model)
            self.counter = 0
            self.test_score = test

    def save_checkpoint(self, val, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            if self.use_loss:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val:.6f}).  Saving model ...')
            else:
                print(
                    f'Validation score increased ({self.val_score_max:.6f} --> {val:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)
        if self.use_loss:
            self.val_loss_min = val
        else:
            self.val_score_max = val
