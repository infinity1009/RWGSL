import sys
import torch
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from normalization import row_normalize

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_hetero(dataset='acm'):
    dataset_dir = f'data/{dataset}/'
    with open(dataset_dir + "node_features.pkl", "rb") as f:
        features = pkl.load(f)
    if sp.issparse(features):
        features = features.todense()
    features = torch.FloatTensor(np.asarray(features))
    with open(dataset_dir + "edges.pkl", "rb") as f:
        edges = pkl.load(f)
    with open(dataset_dir + "labels.pkl", "rb") as f:
        labels = pkl.load(f)
    p_labels = np.load(open(dataset_dir + "p_labels.npy", "rb"))

    adj = np.sum(list(edges.values()))
    
    def get_label():
        '''
        Returns:
            train_index, train_y, val_index, val_y, test_index, test_y: train/val/test index and labels
        '''
        train_index = torch.from_numpy(np.array(labels[0])[:, 0]).type(torch.LongTensor)
        train_y = torch.from_numpy(np.array(labels[0])[:, 1]).type(torch.LongTensor)
        val_index = torch.from_numpy(np.array(labels[1])[:, 0]).type(torch.LongTensor)
        val_y = torch.from_numpy(np.array(labels[1])[:, 1]).type(torch.LongTensor)
        test_index = torch.from_numpy(np.array(labels[2])[:, 0]).type(torch.LongTensor)
        test_y = torch.from_numpy(np.array(labels[2])[:, 1]).type(torch.LongTensor)
        return train_index, train_y, val_index, val_y, test_index, test_y
    
    train_index, train_y, val_index, val_y, test_index, test_y = get_label()
    labels = torch.from_numpy(np.zeros_like(p_labels[:, 0])).type(torch.LongTensor)
    labels[train_index] = train_y 
    labels[val_index] = val_y 
    labels[test_index] = test_y
    
    return adj, features, labels, p_labels, train_index, val_index, test_index

def load_ogb_data(name, root="data", noise="none", drop_rate=0.6, add_rate=0.2):
    if noise == "add":
        adj = sp.load_npz(f"./data/random/{name}/add_{str(add_rate)}.npz")
        p_labels = np.load(open(f"./data/{name}/{name}_{noise}_p_labels.npy", "rb"))
    elif noise == "delete":
        adj = sp.load_npz(f"./data/random/{name}/drop_{str(drop_rate)}.npz")
        p_labels = np.load(open(f"./data/{name}/{name}_{noise}_p_labels.npy", "rb"))
    else:
        adj = sp.load_npz(f"./data/{name}/original_adj.npz")
        p_labels = np.load(open(f"./data/{name}/{name}_{noise}_p_labels.npy", "rb"))

    name_map = {"ogbn_products": "ogbn-products", "ogbn_arxiv": "ogbn-arxiv"}
    name = name_map[name]
    dataset = PygNodePropPredDataset(name, root)
    splits = dataset.get_idx_split()
    train_index, val_index, test_index = (
        splits["train"],
        splits["valid"],
        splits["test"],
    )

    data = dataset[0]
    evaluator = Evaluator(name)

    return (
        adj,
        data.x,
        data.y,
        p_labels,
        train_index,
        val_index,
        test_index,
        evaluator,
    )

def load_citation_data(dataset_str="cora", drop_rate=0, add_rate=0, mask_feat_rate=0, label_per_class=20):
    """
    Load Citation Networks Datasets.
    """
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), "rb") as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding="latin1"))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == "citeseer":
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features = torch.FloatTensor(np.array(features.todense())).float()

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    if dataset_str == "citeseer":
        features = row_normalize(features)

    features = torch.FloatTensor(features).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    p_labels = np.load(open(f"data/{dataset_str}/{dataset_str}_none_p_labels.npy", "rb"))

    adj = sp.coo_matrix(adj)
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj = adj.tocsr()

    return adj, features, labels, p_labels, idx_train, idx_val, idx_test