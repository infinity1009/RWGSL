import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv

class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x):
        return self.W(x)

class FeedForwardNet(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            for _ in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
            self.layers.append(nn.Linear(hidden, out_feats))
        if self.n_layers > 1:
            self.prelu = nn.PReLU()
            self.dropout = dropout
        
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.n_layers - 1:
                x = F.dropout(self.prelu(x), p=self.dropout, training=self.training)
        return x

class SIGN(nn.Module):
    def __init__(
        self,
        in_feats,
        hidden,
        out_feats,
        num_hops,
        n_layers,
        dropout,
        input_drop,
    ):
        super(SIGN, self).__init__()
        self.dropout = dropout
        self.prelu = nn.PReLU()
        self.inception_ffs = nn.ModuleList()
        self.input_drop = input_drop
        for _ in range(num_hops):
            self.inception_ffs.append(
                FeedForwardNet(in_feats, hidden, hidden, n_layers, dropout)
            )
        self.project = FeedForwardNet(
            num_hops * hidden, hidden, out_feats, n_layers, dropout
        )

    def forward(self, feats):
        feats = [F.dropout(feat, p=self.input_drop, training=self.training) for feat in feats]
        hidden = []
        for feat, ff in zip(feats, self.inception_ffs):
            hidden.append(ff(feat))
        out = self.project(F.dropout(self.prelu(torch.cat(hidden, dim=-1)), p=self.dropout, training=self.training))
        return out

    def reset_parameters(self):
        for ff in self.inception_ffs:
            ff.reset_parameters()
        self.project.reset_parameters()

class GCNConv(nn.Module):
    """
    A Graph Convolution Layer (GCN)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, out_features, bias=bias)
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = self.W(input)
        output = torch.spmm(adj, support)
        return output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers=2, batch_norm=False):
        super(GCN, self).__init__()
        self.gcs = nn.ModuleList()
        self.gcs.append(GCNConv(nfeat, nhid))
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bns = nn.ModuleList()
            self.bns.append(nn.BatchNorm1d(nhid))
        for _ in range(nlayers-2):
            self.gcs.append(GCNConv(nhid, nhid))
            if self.batch_norm:
                self.bns.append(nn.BatchNorm1d(nhid))
        self.gcs.append(GCNConv(nhid, nclass))
        self.dropout = dropout

    def forward(self, x, adj, use_relu=True):
        for i, conv in enumerate(self.gcs[:-1]):
            x = conv(x, adj)
            if self.batch_norm:
                x = self.bns[i](x)
            if use_relu:
                x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcs[-1](x, adj)
        return x

class GraphSAGE(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout=0.5, n_layers=2, activation=F.relu):
        super(GraphSAGE, self).__init__()
        self.gcs = nn.ModuleList()
        self.gcs.append(SAGEConv(n_feat, n_hid))
        self.n_layers = n_layers
        for _ in range(n_layers-2):
            self.gcs.append(SAGEConv(n_hid, n_hid))
        self.gcs.append(SAGEConv(n_hid, n_class))
        self.dropout = dropout
        self.activation = activation

    def reset_parameters(self):
        for conv in self.gcs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.gcs):
            x = conv(x, edge_index)
            if i != self.n_layers - 1:
                x = self.activation(x)
                x = F.dropout(x, self.dropout, training=self.training)
        
        return x
    
    def inference(self, x_all, eval_loader, device):
        for i in range(self.n_layers):
            xs = []
            for batch in eval_loader:
                x = x_all[batch.n_id].to(device)
                edege_index = batch.edge_index.to(device)
                x = self.gcs[i](x, edege_index)
                x = x[:batch.batch_size]
                if i != self.n_layers - 1:
                    x = self.activation(x)
                xs.append(x.cpu())
            x_all = torch.cat(xs, dim=0)
        
        return x_all

class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, n_heads, n_layers=2, dropout=0.6, activation=F.elu, attn_dropout=0.6, batch_norm=False):
        super(GAT, self).__init__()
        self.gcs = nn.ModuleList()       
        self.gcs.append(GATConv(n_feat, n_hid // n_heads[0], n_heads[0], dropout=attn_dropout))
        self.n_layers = n_layers
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bns = nn.ModuleList()
            self.bns.append(nn.BatchNorm1d(n_hid))
        for i in range(n_layers-2):
            self.gcs.append(GATConv(n_hid, n_hid // n_heads[i + 1], n_heads[i + 1], dropout=attn_dropout))
            if self.batch_norm:
                self.bns.append(nn.BatchNorm1d(n_hid))
        self.gcs.append(GATConv(n_hid, n_class, n_heads[-1], concat=False, dropout=attn_dropout))
        self.dropout = dropout
        self.activation = activation

    def reset_parameters(self):
        for conv in self.gcs:
            conv.reset_parameters()
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, block):
        repr = x
        if isinstance(block, torch.Tensor):
            block = [block]
        if len(block) == self.n_layers:
            for i in range(self.n_layers-1):
                root_size = block.root_size(i)
                root_repr = repr[:root_size]
                repr = self.gcs[i]((repr, root_repr), block[i])
                if self.batch_norm:
                    repr = self.bns[i](repr)
                repr = self.activation(repr)
                repr = F.dropout(repr, self.dropout, training=self.training)
            root_size = block.root_size(-1)
            root_repr = repr[:root_size]
            repr = self.gcs[-1]((repr, root_repr), block[-1])
        elif len(block) == 1:
            for i in range(self.n_layers-1):
                repr = self.gcs[i](repr, block[0])
                if self.batch_norm:
                    repr = self.bns[i](repr)
                repr = self.activation(repr)
                repr = F.dropout(repr, self.dropout, training=self.training)
            repr = self.gcs[-1](repr, block[0])
        else:
            raise ValueError('The sampling layer must be equal to GNN layer.')
        
        return F.log_softmax(repr, dim=-1)