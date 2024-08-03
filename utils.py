import math
import torch
import platform
import numpy as np
from time import perf_counter
import numpy.random as random
import torch.nn.functional as F

from models import SGC, GCN, SIGN, GraphSAGE, GAT
from speedup import csr_sparse_dense_matmul
from normalization import fetch_normalization
from torch_geometric.utils import from_scipy_sparse_matrix

def set_seed(seed, cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.col, sparse_mx.row)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def calculate_similarity(indptr, indices, raw_features, smooth_labels, u, v, sim_types):
    score, cnt = 0., 0
    if (sim_types & 1) == 1:
        score += calculate_feature_similarity(raw_features, u, v)
        cnt += 1 
    if ((sim_types >> 1) & 1) == 1:
        score += calculate_feature_similarity(smooth_labels, u, v)
        cnt += 1
    if ((sim_types >> 2) & 1) == 1:
        score += calculate_Jaccard_coefficient(indptr, indices, u, v)
        cnt += 1
    if ((sim_types >> 3) & 1) == 1:
        score += calculate_AAI(indptr, indices, u, v)
        cnt += 1

    return score / cnt

def calculate_feature_similarity(features, u, v):
    similarity = torch.sum(torch.mul(features[u], features[v])) / (
        (torch.norm(features[u], p=2) * torch.norm(features[v], p=2))
    )

    return similarity

def calculate_Jaccard_coefficient(indptr, indices, u, v):
    a = indices[indptr[u] : indptr[u + 1]]
    b = indices[indptr[v] : indptr[v + 1]]
    common_neighbor = len(set(a).intersection(set(b)))
    similarity = common_neighbor / (len(a) + len(b) - common_neighbor)
    
    return similarity

def calculate_AAI(indptr, indices, u, v):
    a = indices[indptr[u] : indptr[u + 1]]
    b = indices[indptr[v] : indptr[v + 1]]
    common_neighbor = set(a).intersection(set(b))
    similarity = 0
    for node in common_neighbor:
        deg = indptr[node + 1] - indptr[node]
        similarity += 1. / math.log(deg, 2)
    
    return similarity

def sgc_precompute(features, adj, degree, model="SGC", alpha=0.05):
    if model == "SSGC":
        t = perf_counter()
        emb = alpha * features
        for _ in range(degree):
            if platform.system() == "Linux":
                features = csr_sparse_dense_matmul(adj, features)
            else:
                features = adj.dot(features)
            emb = emb + (1 - alpha) * features / degree

        return torch.FloatTensor(emb), perf_counter() - t
    
    elif model == "SGC":
        t = perf_counter()
        for _ in range(degree):
            if platform.system() == "Linux":
                features = csr_sparse_dense_matmul(adj, features)
            else:
                features = adj.dot(features)

        return torch.FloatTensor(features), perf_counter() - t
    
    elif model == "LP":
        n_classes = features.max() + 1
        if features.ndim == 1:
            features = np.eye(n_classes)[features]
        features = features.astype(np.float32)
        t = perf_counter()
        for _ in range(degree):
            if platform.system() == "Linux":
                smooth_features = csr_sparse_dense_matmul(adj, features)
            else:
                smooth_features = adj.dot(features)
            features = (1 - alpha) * smooth_features + alpha * features

        return torch.FloatTensor(features), perf_counter() - t
    
    else:
        raise NotImplementedError


def sign_precompute(features, adj, num_layers):
    t = perf_counter()
    processed_list = []
    for _ in range(num_layers):
        if platform.system() == "Linux":
            features = csr_sparse_dense_matmul(adj, features)
        else:
            features = adj.dot(features)
        processed_list.append(torch.FloatTensor(features))
    precompute_time = perf_counter() - t
    
    return processed_list, precompute_time

def get_smooth_labels(adj, p_labels, lp_num_layers, lp_alpha):
    adj_normalizer = fetch_normalization("AugNormAdj")
    adj = adj_normalizer(adj)
    smooth_labels = sgc_precompute(p_labels, adj.tocsr(), lp_num_layers, "LP", lp_alpha)[0]

    return smooth_labels

def get_smooth_features(adj, features, degree, normalization="AugNormAdj", trans_type="SGC", alpha=0.05):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    if trans_type == "SSGC":
        embedding = sgc_precompute(features.numpy(), adj.tocsr(), degree, trans_type, alpha)[0]
    else:
        embedding = sgc_precompute(features.numpy(), adj.tocsr(), degree)[0]
    
    return embedding


def visual_feature_similarity(adj, features):
    distribution = [0] * 10
    total_edge = 0
    adj = adj.todense()
    n = adj.shape[0]
    for i in range(n - 1):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                total_edge += 1
                x = calculate_feature_similarity(features, i, j)
                ind = int(x * 10)
                if ind == 10:
                    distribution[-1] += 1
                else:
                    distribution[ind] += 1
    
    return distribution, total_edge

def precompute(args, adj, features):
    if args.model in ["SGC", "SSGC"]:
        adj_normalizer = fetch_normalization(args.normalization)
        adj_1 = adj_normalizer(adj)
        adj_tensor_1 = sparse_mx_to_torch_sparse_tensor(adj_1).float()
        processed_features, precompute_time = sgc_precompute(
            features.numpy(), adj_1.tocsr(), args.model_degree, args.model, args.alpha
        )

    elif args.model == "SIGN":
        adj_normalizer = fetch_normalization(args.normalization)
        adj_1 = adj_normalizer(adj)
        adj_tensor_1 = sparse_mx_to_torch_sparse_tensor(adj_1).float()
        processed_features, precompute_time = sign_precompute(
            features.numpy(), adj_1.tocsr(), args.model_degree
        )
    
    else:
        if args.model in ["GCN", "GCN-bn"]:
            adj_normalizer = fetch_normalization(args.normalization)
            adj_1 = adj_normalizer(adj)
            adj_tensor_1 = sparse_mx_to_torch_sparse_tensor(adj_1).float()

        elif args.model in ["SAGE", "SAGE-bn", "GAT"]:
            adj_tensor_1 = from_scipy_sparse_matrix(adj)[0]

        elif args.model == "GCN-hetero":
            adj_tensor_1 = torch.from_numpy(adj.todense()).type(torch.FloatTensor)
            adj_tensor_1 = F.normalize(adj_tensor_1, dim=1, p=1)
        
        processed_features, precompute_time = torch.FloatTensor(features), 0

    return processed_features, precompute_time, adj_tensor_1

def get_model(args, nfeat, nclass):
    if args.model in ["SGC", "SSGC"]:
        model = SGC(nfeat, nclass)
    elif args.model == "SIGN":
        model = SIGN(nfeat, args.hidden_channels, nclass, args.model_degree, args.ff_layer, args.dropout, args.input_drop)
    elif args.model == "GCN-bn":
        model = GCN(nfeat, args.hidden_channels, nclass, args.dropout, 3, True)
    elif args.model == "SAGE":
        model = GraphSAGE(nfeat, args.hidden_channels, nclass, args.dropout, args.num_layers)
    elif args.model == "GAT":
        n_heads = [int(head) for head in args.nheads.split(",")]
        model = GAT(nfeat, args.hidden_channels, nclass, n_heads=n_heads, dropout=args.dropout)
    elif args.model.startswith("GCN"):
        model = GCN(nfeat, args.hidden_channels, nclass, args.dropout)

    return model

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience,
        save_model=False,
        verbose=False,
        delta=0.0,
        save_path="./",
        use_loss=True,
    ):
        self.patience = patience
        self.save_model = save_model
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_score_max = 0.0
        self.test_score = 0.0
        self.delta = delta
        self.save_path = save_path + "checkpoint.pt"
        self.use_loss = use_loss
        self.preds = None

    def __call__(self, val, test, model, preds=None):
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
                    f"Validation loss decreased ({self.val_loss_min:.6f} --> {val:.6f}).  Saving model ..."
                )
            else:
                print(
                    f"Validation score increased ({self.val_score_max:.6f} --> {val:.6f}).  Saving model ..."
                )
        torch.save(model.state_dict(), self.save_path)
        if self.use_loss:
            self.val_loss_min = val
        else:
            self.val_score_max = val
