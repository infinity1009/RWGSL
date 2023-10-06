import numpy as np
import scipy.sparse as sp
import torch

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalized_adjacency(adj): 
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo() # sp格式的稀疏矩阵

def aug_normalized_adjacency(adj): 
   adj = adj + sp.eye(adj.shape[0]) #A+I
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo() # sp格式的稀疏矩阵

def aug_normalized_adjacency2(adj):
   adj =  torch.eye(adj.size()[0]) + adj  #A+I
   #adj = sp.eye(adj.size()[0]) + adj
   row_sum = torch.sum(adj, 1)
   d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
   d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
   return torch.mm(torch.mm(d_mat_inv_sqrt, adj),d_mat_inv_sqrt) # tensor格式的稀疏张量

def create_sparse_I_tensor(n):
    index = torch.tensor([[i for i in range(n)],[i for i in range(n)]])
    values = torch.tensor([1 for i in range(n)])
    shape = torch.Size((n,n))
    return torch.sparse.FloatTensor(index,values,shape)


def fetch_normalization(type):
   switcher = {
        'NormAdj': normalized_adjacency,
       'AugNormAdj': aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
   }
   func = switcher.get(type, lambda: "Invalid normalization technique.")
   return func

def fetch_normalization2(type):
   switcher = {
       'AugNormAdj': aug_normalized_adjacency2,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
   }
   func = switcher.get(type, lambda: "Invalid normalization technique.")
   return func
def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
