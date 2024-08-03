import torch
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool

def one_layer_sampling(adj, tgt_nodes: list, layer_size: int):
    subgraph_adj = adj[tgt_nodes, :]
    neis = np.nonzero(np.sum(subgraph_adj, axis=0))[1]
    layer_size = min(len(neis), layer_size)
    local_nids = np.random.choice(np.arange(np.size(neis)), layer_size, False)
    source_nodes = neis[local_nids]
    
    return source_nodes

def layer_wise_sampling(adj: sp.csr_matrix, num_samples: list, node_id: int):
    sample_list = []
    num_layers = len(num_samples)
    cur_tgt_nodes = [node_id]
    for layer_index in range(num_layers):
        cur_src_nodes = one_layer_sampling(adj, cur_tgt_nodes, num_samples[layer_index])
        cur_tgt_nodes = cur_src_nodes
        sample_list.extend(cur_src_nodes)

    sample_list = list(set(sample_list)) # remove replicated nodes
    return sample_list

class LayerWiseRandomSample:
    def __init__(self, adj: sp.csr_matrix, num_samples: list):
        self.adj = adj
        self.num_samples = num_samples
        self.num_layers = len(self.num_samples)

    def __call__(self, node_id: int):
        sample_list = []
        cur_tgt_nodes = [node_id]
        for layer_index in range(self.num_layers):
            cur_src_nodes = one_layer_sampling(self.adj, cur_tgt_nodes, self.num_samples[layer_index])
            cur_tgt_nodes = cur_src_nodes
            sample_list.extend(cur_src_nodes)

        sample_list = list(set(sample_list)) # remove replicated nodes
        return sample_list


def layer_wise_sample(adj: sp.csr_matrix, num_samples: list, pool_num: int):
    n = adj.shape[0]
    p = Pool(pool_num)
    layer_wise_sample_list = p.map(LayerWiseRandomSample(adj, num_samples), [i for i in range(n)])
    p.close()
    p.join()
    
    return layer_wise_sample_list

def isolate_node(adj):
    degree = torch.sum(adj, axis=0)
    z = torch.zeros(adj.size()[0])
    isolate_nodes = torch.sum((degree == z))
    return isolate_nodes