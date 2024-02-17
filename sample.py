import torch
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool


class RandomSample:
    def __init__(self, adj: sp.csr_matrix, s1: int):
        self.indptr = adj.indptr
        self.indices = adj.indices
        self.deg_1 = s1

    def __call__(self, j: int):
        deg = self.indptr[j + 1] - self.indptr[j]
        if deg == 0:
            return np.array([])

        perm = np.arange(deg)
        np.random.shuffle(perm)
        perm_indices = self.indices[self.indptr[j] : self.indptr[j + 1]][perm]

        nums = min(deg, self.deg_1)

        return perm_indices[:nums]


def random_sample(adj: sp.csr_matrix, s1: int):
    n = adj.shape[0]
    p = Pool(3)
    sample_list = p.map(RandomSample(adj, s1), [i for i in range(n)])
    p.close()
    p.join()
    
    return sample_list


def random_walk(walk_length, indptr, indices, start_node):
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = indices[indptr[cur] : indptr[cur + 1]]
        if len(cur_nbrs) > 0:
            walk.append(np.random.choice(cur_nbrs, 1).item())
        else:
            break
    
    return walk

def isolate_node(adj):
    degree = torch.sum(adj, axis=0)
    z = torch.zeros(adj.size()[0])
    isolate_nodes = torch.sum((degree == z))
    return isolate_nodes
