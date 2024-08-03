import torch
import numpy as np
import scipy.sparse as sp
from typing import List
from multiprocessing import Pool

from sample import layer_wise_sampling
from wrapper import random_walks_sample
from utils import calculate_Jaccard_coefficient, calculate_feature_similarity

def modify_structure(adj, raw_features, smooth_labels, args):
    random_walk_sample_list = random_walks_sample(adj, args.walk_len)[0]
    adj = modify(
        adj,
        args.num_samples,
        random_walk_sample_list,
        raw_features,
        smooth_labels,
        low_threshold=args.low,
        high_threshold=args.high,
        first_coe=args.first_coe,
        second_coe=args.second_coe,
        third_coe=args.third_coe,
        pool_num=args.pool_num
    )
    return adj

class Modifier:
    def __init__(
        self,
        adj: sp.csr_matrix,
        num_samples: list,
        random_walk_sample_list: np.ndarray,
        raw_features: torch.FloatTensor,
        smooth_labels: torch.FloatTensor,
        low_threshold: float,
        high_threshold: float,
        first_coe: float,
        second_coe: float,
        third_coe: float,
    ):
        self.adj = adj
        self.indptr = adj.indptr
        self.indices = adj.indices
        self.num_samples = num_samples
        self.random_walk_sample_list = random_walk_sample_list
        self.raw_features = raw_features
        self.smooth_labels = smooth_labels
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.first_coe = first_coe
        self.second_coe = second_coe
        self.third_coe = third_coe

    def __call__(self, current_node: int):
        """
        local exploitation (layer-wise sampling) and global exploration (long length random walk)
        """
        start, end = self.indptr[current_node], self.indptr[current_node+1]
        full_neighbors = list(self.indices[start: end])
        length = len(full_neighbors)   
        if length == 0:
            return None, None
        
        sampled_nodes = layer_wise_sampling(self.adj, self.num_samples, current_node)

        d_row, d_col = [], []
        a_row, a_col = [], []

        sim_list = []
        for i in range(len(sampled_nodes)):
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
        for i, node in enumerate(sampled_nodes):
            if node in full_neighbors and sim_list[i] < self.low_threshold * mean_sim:
                d_row.append(current_node)
                d_col.append(node)
            if node not in full_neighbors and sim_list[i] > self.high_threshold * mean_sim:
                a_row.append(current_node)
                a_col.append(node)
        
        walk_list = self.random_walk_sample_list[current_node, 1:] # remove the current_node itself
        
        if len(walk_list) == 0:
            return np.vstack([d_row, d_col]), np.vstack([a_row, a_col])

        sim_list = []
        raw_sim_list = []
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
            raw_sim_list.append(raw_similarity)

        mean_sim = np.mean(sim_list)
        
        for i, node in enumerate(walk_list):
            if node in full_neighbors and node not in d_col and sim_list[i] < self.low_threshold * mean_sim:
                d_row.append(current_node)
                d_col.append(node)
            if max(sim_list[i], raw_sim_list[i]) > self.high_threshold * mean_sim:
                a_row.append(current_node)
                a_col.append(node)
        
        return np.vstack([d_row, d_col]), np.vstack([a_row, a_col])


def modify(
    adj: sp.csr_matrix,
    num_samples: list,
    random_walk_sample_list: np.ndarray,
    raw_features: torch.FloatTensor,
    smooth_labels: torch.FloatTensor,
    low_threshold: float,
    high_threshold: float,
    first_coe: float,
    second_coe: float,
    third_coe: float,
    pool_num: int
):
    n = adj.shape[0]
    adj_lil = adj.tolil(copy=True)
    
    def get_mod(modify_list: List):
        for mod_pair in modify_list:
            d_edges = mod_pair[0]
            if d_edges is None:
                continue
            num_edges = d_edges.shape[1]
            for i in range(num_edges):
                adj_lil[d_edges[0][i], d_edges[1][i]] = 0
                adj_lil[d_edges[1][i], d_edges[0][i]] = 0
            a_edges = mod_pair[1]
            if a_edges is None:
                continue
            num_edges = a_edges.shape[1]
            for i in range(num_edges):
                adj_lil[a_edges[0][i], a_edges[1][i]] = 1
                adj_lil[a_edges[1][i], a_edges[0][i]] = 1

    def get_error(value):
        print(f"error message: {value}")

    p = Pool(pool_num)
    p.map_async(Modifier(adj, num_samples, random_walk_sample_list, raw_features, smooth_labels, \
                 low_threshold, high_threshold, first_coe, second_coe, third_coe), \
                    [i for i in range(n)], callback=get_mod, error_callback=get_error)
    p.close() # close the process pool
    p.join() # the main process waits for the completions of sub-processes

    return adj_lil