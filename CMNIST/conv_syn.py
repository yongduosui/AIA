import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from torch_geometric.utils import degree
from torch_scatter import scatter_add
from torch_geometric.nn.inits import reset
import math
import pdb

class GINConv(MessagePassing):
    def __init__(self, in_dim, emb_dim):
        
        super(GINConv, self).__init__(aggr = "add")
        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, 2*emb_dim),
                                       torch.nn.BatchNorm1d(2*emb_dim), 
                                       torch.nn.ReLU(), 
                                       torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_weight=None):
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_weight=edge_weight))
        return out
    def message(self, x_j, edge_weight=None):
        if edge_weight is not None:
            mess = F.relu(x_j * edge_weight)
        else:
            mess = F.relu(x_j)
        return mess

    def update(self, aggr_out):
        return aggr_out


class GNNSynEncoder(torch.nn.Module):
    def __init__(self, num_layer, in_dim, emb_dim, dropout_rate=0.5):
       
        super(GNNSynEncoder, self).__init__()
        self.num_layer = num_layer
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.dropout_rate = dropout_rate
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList([nn.ReLU() for _ in range(num_layer - 1)])
        self.batch_norm1 = nn.BatchNorm1d(emb_dim)
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layer - 1)])
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.dropouts = nn.ModuleList([nn.Dropout(self.dropout_rate) for _ in range(num_layer - 1)])
        self.conv1 = GINConv(in_dim, emb_dim)
        self.convs = nn.ModuleList([GINConv(emb_dim, emb_dim) for _ in range(num_layer - 1)])

    def forward(self, x, edge_index, node_adv=None, edge_adv=None):
        
        if node_adv is not None:
            x = x * node_adv
        post_conv = self.batch_norm1(self.conv1(x, edge_index, edge_adv))
        if self.num_layer > 1:
            post_conv = self.relu1(post_conv)
            post_conv = self.dropout1(post_conv)
        
        for i, (conv, batch_norm, relu, dropout) in enumerate(zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = batch_norm(conv(post_conv, edge_index, edge_adv))
            if i != len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)
        return post_conv


class GraphSynMasker(torch.nn.Module):

    def __init__(self, num_layer, in_dim, emb_dim, dropout_rate=0.5):
        super(GraphSynMasker, self).__init__()

        self.gnn_encoder = GNNSynEncoder(num_layer, in_dim, emb_dim, dropout_rate)
        self.edge_att_mlp = nn.Linear(emb_dim * 2, 1)
        self.node_att_mlp = nn.Linear(emb_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gnn_encoder)
        reset(self.edge_att_mlp)
        reset(self.node_att_mlp)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        node_rep = self.gnn_encoder(x, edge_index)
        size = batch[-1].item() + 1
        row, col = edge_index
        edge_rep = torch.cat([node_rep[row], node_rep[col]], dim=-1)
        node_key = torch.sigmoid(self.node_att_mlp(node_rep))
        edge_key = torch.sigmoid(self.edge_att_mlp(edge_rep))

        node_key_num, node_env_num, non_zero_node_ratio = self.reg_mask(node_key, batch, size)
        edge_key_num, edge_env_num, non_zero_edge_ratio = self.reg_mask(edge_key, batch[edge_index[0]], size)

        self.non_zero_node_ratio = non_zero_node_ratio
        self.non_zero_edge_ratio = non_zero_edge_ratio

        output = {"node_key": node_key, "edge_key": edge_key,
                  "node_key_num": node_key_num, "node_env_num": node_env_num,
                  "edge_key_num": edge_key_num, "edge_env_num": edge_env_num}
        return output
    
    def reg_mask(self, mask, batch, size):

        key_num = scatter_add(mask, batch, dim=0, dim_size=size)
        env_num = scatter_add((1 - mask), batch, dim=0, dim_size=size)
        non_zero_mask = scatter_add((mask > 0).to(torch.float32), batch, dim=0, dim_size=size) 
        all_mask = scatter_add(torch.ones_like(mask).to(torch.float32), batch, dim=0, dim_size=size)
        non_zero_ratio = non_zero_mask / all_mask
        return key_num + 1e-8, env_num + 1e-8, non_zero_ratio

