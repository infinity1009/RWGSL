import torch
import numpy as np
import scipy.sparse as sp
from random import shuffle
from typing import List, Union
from multiprocessing import Pool

from sample import random_sample, random_walk
from utils import calculate_Jaccard_coefficient, calculate_feature_similarity

def modify_structure(adj, raw_features, smooth_labels, args):
    if args.random_sample:
        sample_list = random_sample(adj, s1=args.separate_1)
    else:
        sample_list = None
    adj = modify(
        adj,
        raw_features,
        smooth_labels,
        sample_list,
        topk=args.topk,
        low_threshold=args.low,
        high_threshold=args.high,
        walk_len=args.walk_len,
        first_coe=args.first_coe,
        second_coe=args.second_coe,
        third_coe=args.third_coe,
        fourth_coe=args.fourth_coe,
        pool_num=args.pool_num
    )
    return adj

class Modifier:
    def __init__(
        self,
        indptr: np.ndarray,
        indices: np.ndarray,
        raw_features: torch.FloatTensor,
        smooth_labels: torch.FloatTensor,
        sample_list: Union[List, None],
        topk: float,
        low_threshold: float,
        high_threshold: float,
        walk_len: int,
        first_coe: float,
        second_coe: float,
        third_coe: float,
        fourth_coe: float
    ):
        self.indptr = indptr
        self.indices = indices
        self.raw_features = raw_features
        self.smooth_labels = smooth_labels
        self.sample_list = sample_list
        self.topk = topk
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.walk_len = walk_len
        self.first_coe = first_coe
        self.second_coe = second_coe
        self.third_coe = third_coe
        self.fourth_coe = fourth_coe

    def __call__(self, current_node: int):
        start, end = self.indptr[current_node], self.indptr[current_node+1]
        full_neighbors = list(self.indices[start: end])
        if self.sample_list is not None:
            sampled_nodes = self.sample_list[current_node]
        else:
            sampled_nodes = list(self.indices[start: end])
        
        length = len(sampled_nodes)   
        
        if length == 0:
            return None, None

        start_list = []
        d_row, d_col = [], []
        a_row, a_col = [], []

        sim_list = []
        for i in range(length):
            node = sampled_nodes[i]
            raw_similarity = calculate_feature_similarity(self.raw_features, current_node, node)
            jaccard_similarity = calculate_Jaccard_coefficient(self.indptr, self.indices, current_node, node)
            label_similarity = calculate_feature_similarity(self.smooth_labels, current_node, node)
            sign_raw = 1 if raw_similarity >= 0 else -1
            sim_score = (
                sign_raw * pow(sign_raw * raw_similarity, self.first_coe)
                + pow(jaccard_similarity, self.second_coe)
                + pow(label_similarity, self.third_coe)
            ) / 3
            sim_list.append(sim_score)
        
        mean_sim = np.mean(sim_list)
        for i in range(length):
            if sim_list[i] < self.low_threshold * mean_sim:
                d_row.append(current_node)
                d_col.append(sampled_nodes[i])
            else:
                start_list.append(sampled_nodes[i])
        
        if self.topk > 0 and len(start_list) >= self.topk:
            shuffle(start_list)
            start_list = start_list[: self.topk]
        
        walk_list = []
        for start_node in start_list:
            walk = random_walk(self.walk_len, self.indptr, self.indices, start_node)
            walk_list.extend(walk)

        walk_list = set(walk_list)
        walk_list.discard(current_node)
        walk_list = list(walk_list)
        
        if len(walk_list) == 0:
            return np.vstack([d_row, d_col]), None 

        sim_list = []
        for node in walk_list:
            raw_similarity = calculate_feature_similarity(self.raw_features, current_node, node)
            jaccard_similarity = calculate_Jaccard_coefficient(self.indptr, self.indices, current_node, node)
            label_similarity = calculate_feature_similarity(self.smooth_labels, current_node, node)
            sign_raw = 1 if raw_similarity >= 0 else -1   
            sim_score = (
                sign_raw * pow(sign_raw * raw_similarity, self.first_coe)
                + pow(jaccard_similarity, self.second_coe)
                + pow(label_similarity, self.third_coe)
            ) / 3 
            sim_list.append(sim_score)

        mean_sim = np.mean(sim_list)
        
        for i, node in enumerate(walk_list):
            if node in full_neighbors and node not in d_col and sim_list[i] < self.low_threshold * mean_sim:
                d_row.append(current_node)
                d_col.append(node)
            if max(sim_list[i], raw_similarity) > self.high_threshold * mean_sim:
                a_row.append(current_node)
                a_col.append(node)
        
        return np.vstack([d_row, d_col]), np.vstack([a_row, a_col])


def modify(
    adj: sp.csr_matrix,
    raw_features: torch.FloatTensor,
    smooth_labels: torch.FloatTensor,
    sample_list: List,
    topk: int,
    low_threshold: float,
    high_threshold: float,
    walk_len: int,
    first_coe: float,
    second_coe: float,
    third_coe: float,
    fourth_coe: float,
    pool_num: int
):
    n = adj.shape[0]
    indptr, indices = adj.indptr, adj.indices
    adj = adj.tolil()
    
    def get_mod(modify_list: List):
        for mod_pair in modify_list:
            d_edges = mod_pair[0]
            if d_edges is None:
                continue
            num_edges = d_edges.shape[1]
            for i in range(num_edges):
                adj[d_edges[0][i], d_edges[1][i]] = 0
                adj[d_edges[1][i], d_edges[0][i]] = 0
            a_edges = mod_pair[1]
            if a_edges is None:
                continue
            num_edges = a_edges.shape[1]
            for i in range(num_edges):
                adj[a_edges[0][i], a_edges[1][i]] = 1
                adj[a_edges[1][i], a_edges[0][i]] = 1

    def get_error(value):
        print(f"error message: {value}")

    p = Pool(pool_num)
    p.map_async(Modifier(indptr, indices, raw_features, smooth_labels, sample_list, \
                 topk, low_threshold, high_threshold, walk_len, first_coe, second_coe, third_coe, fourth_coe), \
                    [i for i in range(n)], callback=get_mod, error_callback=get_error)
    p.close() # close the process pool
    p.join() # the main process waits for the completions of sub-processes

    return adj