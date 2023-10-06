from random import shuffle, randint, choice
import scipy.sparse as sp
import numpy as np
import torch


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.col, sparse_mx.row)).astype(np.int64)) 
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def random_sample(adj, s1: int, s2: int, m: float):
    n = adj.shape[0]
    index = adj.nonzero() 
    row_index = index[0]  
    column_index = index[1] 
    slice_index = []
    len_row = len(column_index)
    slice_index.extend([0] * (row_index[0]+1))
    for i in range(1, len_row): 
        if row_index[i] != row_index[i-1]:  
            slice_index.extend([i] * (row_index[i]-row_index[i-1]))
    slice_index.extend([len_row] * (n-row_index[-1]))
    
    assert len(slice_index) == n + 1, 'slice_index has bugs!'
    sample_dict = {}

    neighbor_dict = create_neighbor_dict(index, slice_index) 

    for j in range(n):
        d = slice_index[j+1] - slice_index[j]  
        if d == 0: 
            continue
        current_node = j 

        a = np.arange(d)
        np.random.shuffle(a)
        column_index[slice_index[j]:slice_index[j+1]] = column_index[slice_index[j]:slice_index[j+1]][a]
        
        if d <= s1:
            nums = d  
        elif s1 < d < s2:
            nums = int(s1 + m * (d - s1))
        else:
            nums = int(s1 + m * (s2 - s1))
        
        sample_dict[current_node] = column_index[slice_index[j]:slice_index[j]+nums]
    
    return sample_dict, neighbor_dict


def random_walk(walk_length, neighbor_dict, start_node): 
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        if cur not in neighbor_dict.keys():
            break
        cur_nbrs = neighbor_dict[cur].tolist()
        if (len(cur_nbrs) > 0):
            walk.append(choice(cur_nbrs))
        else:
            break
    return walk

def create_neighbor_dict(index, slice_index):
    neighbor_dict = {}
    n = len(slice_index)
    for i in range(n-1):
        if slice_index[i] == slice_index[i+1]: 
            neighbor_dict[i] = None
        else:
            neighbor_dict[i] = index[1][slice_index[i]:slice_index[i+1]]

    return neighbor_dict


def calculate_similarity(features, u, v):
    similarity = torch.sum(torch.mul(features[u], features[v]))/(
        (torch.norm(features[u], p=2)*torch.norm(features[v], p=2)))
    return similarity

def calculate_similarity_all(features, u, nodes):
    similarity = torch.sum(torch.mul(features[u], features[nodes]), dim=1)/(
        (torch.norm(features[u], p=2)*torch.norm(features[nodes], p=2, dim=1)))
    return similarity


def calculate_second_similarity(neighbor_dict, u, v):
    a = neighbor_dict[u]
    b = neighbor_dict[v]
    common_neighbor = len(list(set(a).intersection(set(b))))
    similarity = common_neighbor/(len(a)+len(b)-common_neighbor)
    return similarity


def add_edge(adj, u, v): 
    adj[u, v] = 1
    adj[v, u] = 1
    return adj

def add_edges(adj, u, nodes):
    for v in nodes:
        adj[u, v] = adj[v, u] = 1
    return adj

def delete_edges(adj, u, nodes):
    for v in nodes:
        if np.sum(adj[u]).item() == 1 or np.sum(adj[v]).item() == 1:
            continue
        adj[u, v] = 0
        adj[v, u] = 0
    return adj

def delete_edge(adj, u, v):
    if np.sum(adj[u]).item() == 1 or np.sum(adj[v]).item() == 1:
        return adj
    adj[u, v] = 0
    adj[v, u] = 0
    return adj


def modify(adj, raw_features, smooth_labels, sample_dict, neighbor_dict, topk, low_threshold, high_threshold, walk_len, fst, snd, trd):
    adj = sp.lil_matrix(adj)
    for current_node in sample_dict.keys():
        sampled_nodes = sample_dict[current_node].tolist()
        length = len(sampled_nodes)

        start_list = []
        for i in range(length):
            node = sampled_nodes[i]
            raw_similarity = calculate_similarity(raw_features, current_node, node)
            second_similarity = calculate_second_similarity(neighbor_dict, current_node, node)
            label_similarity = calculate_similarity(smooth_labels, current_node, node)
            sign_second = 1 if second_similarity >= 0 else -1
            sign_raw = 1 if raw_similarity >= 0 else -1
            x = (sign_raw * pow(sign_raw * raw_similarity, fst) + sign_second * pow(sign_second * second_similarity, snd) + pow(label_similarity, trd)) / 3
        
            if x < low_threshold:
                adj = delete_edge(adj, current_node, node)
            else:
                start_list.append(node)

        length = len(start_list)
        if length >= topk:
            shuffle(start_list)
            start_list = start_list[:topk]
       
        walk_list = []
        for start_node in start_list:
            walk = random_walk(walk_len, neighbor_dict, start_node)
            walk_list.extend(walk)  
        
        walk_list = set(walk_list)
        walk_list.discard(current_node)
        walk_list = list(walk_list)
        for node in walk_list:
            raw_similarity = calculate_similarity(
                raw_features, current_node, node)
            second_similarity = calculate_second_similarity(
                neighbor_dict, current_node, node)
            label_similarity = calculate_similarity(
                smooth_labels, current_node, node
            )
            sign_second = 1 if second_similarity >= 0 else -1
            sign_raw = 1 if raw_similarity >= 0 else -1
            x = (sign_raw * pow(sign_raw * raw_similarity, fst) + sign_second * pow(sign_second * second_similarity, snd) + pow(label_similarity, trd)) / 3
            if adj[current_node, node] > 0 and x < low_threshold:
                adj = delete_edge(adj, current_node, node)
            elif max(x, raw_similarity) > high_threshold:
                adj = add_edge(adj, current_node, node)

    return adj


def isolate_node(adj):
    degree = torch.sum(adj, axis=0)
    z = torch.zeros(adj.size()[0])
    isolate_nodes = torch.sum((degree == z))
    return isolate_nodes
