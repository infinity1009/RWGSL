import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device or CPU (-1)')
    parser.add_argument('--verbose', action='store_true',
                        default=False, help='print detailed information')
    parser.add_argument('--train_ori', action='store_true',
                        default=False, help='whether to train on original graph')
    parser.add_argument('--pool_num', type=int, default=5, help="pool nums")
    parser.add_argument('--update', type=int, default=3, help="update times")
    parser.add_argument('--walk_len', type=int, default=5,
                        help="length of random walk") 
    parser.add_argument('--drop_rate', type=float, default=0.,
                        help='the probability of randomly drop edges')
    parser.add_argument('--add_rate', type=float, default=0.,
                        help='the probability of randomly add edges')
    parser.add_argument('--mask_feat_rate', type=float, default=0.,
                        help='the probability of randomly mask features')
    parser.add_argument('--label_per_class', type=int, default=20, 
                        help='number of labeled nodes per class')
    parser.add_argument('--high', type=float, default=0.8,
                        help="threshold of adding edge")
    parser.add_argument('--low', type=float, default=0.1,
                        help="threshold of deleting edge")
    parser.add_argument('--topk', type=int, default=-1,
                        help="select the most top k similar neighbor nodes")
    parser.add_argument('--separate_1', type=int, default=5,
                        help="degree threshold 1")
    parser.add_argument('--separate_2', type=int, default=10,
                        help="degree threshold 2")
    parser.add_argument('--coefficient', type=float,
                        default=0.6, help="coefficient in sampling")
    parser.add_argument('--random_sample', action='store_true', default=False, help="whether to do neighborhood sampling")
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')  # 42
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lp_num_layers', type=int,
                        default=3, help='label propagation num layers')
    parser.add_argument('--lp_alpha', type=float,
                        default=0.4, help='label propagation alpha')
    parser.add_argument('--alpha', type=float, default=0.05, 
                        help='SSGC precompute alpha')
    parser.add_argument('--lr', type=float, default=0.2,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1.20e-05,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden_channels', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--model_degree', type=int, default=2,
                        help='propagation degree of model (SGC, SIGN, etc.)')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default="cora",
                        help='Dataset to use.')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='mlp num layers.')
    parser.add_argument('--model', type=str, default="GCN",
                        choices=['SGC', 'SIGN', 'GCN', 'SAGE', 'GAT'],
                        help='model to use.')
    parser.add_argument('--normalization', type=str, default='AugNormAdj',
                        choices=['AugNormAdj', 'NormAdj', 'RowNormAdj'],
                        help='Normalization method for the adjacency matrix.')
    parser.add_argument('--degree', type=int, default=3,
                        help='degree of the approximation.')
    parser.add_argument('--nheads', type=str, default="8,1")
    parser.add_argument('--first_coe', type=float, default=0.5)
    parser.add_argument('--second_coe', type=float, default=0.25)
    parser.add_argument('--third_coe', type=float, default=1)
    parser.add_argument('--fourth_coe', type=float, default=0.5)

    args = parser.parse_args()
    return args
