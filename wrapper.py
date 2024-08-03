import torch
import numpy as np
from torch_geometric import utils

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from walks import get_random_walks

def random_walks_sample(
    csr_matrix,
    length=50,
    sample_rate=1.,
    backtracking=False,
    strict=False,
    window_size=0,
    pad_value=-1,
    rng=None,
):
    if rng is None:
        rng = np.random.mtrand._rand
    if isinstance(rng, int):
        rng = np.random.RandomState(rng)
    csr_matrix = csr_matrix.astype(np.int32)
    return get_random_walks(csr_matrix, length, sample_rate, pad_value, backtracking, strict, window_size, rng)

if __name__ == '__main__':
    edge_index = torch.tensor([[0,0,1,2,2,3,3,4], [1,2,3,4,0,2,0,1]])
    walk_node_index, walk_edge_index, walk_node_id_encoding, walk_node_adj_encoding = random_walks_sample(edge_index, num_nodes=5, length=10, window_size=2)
    print(walk_node_index, walk_node_index.shape)
    print(walk_edge_index, walk_edge_index.shape)
    print(walk_node_id_encoding)
    print(walk_node_adj_encoding)