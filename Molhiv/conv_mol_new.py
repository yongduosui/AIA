import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from MolEncoders import AtomEncoder, BondEncoder
from torch_geometric.utils import degree
from torch_scatter import scatter_add
from torch_geometric.nn.inits import reset
import math
import pdb

nn_act = torch.nn.ReLU() #ReLU()
F_act = F.relu

        
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        
        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), nn_act, torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr, edge_weight=None):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding, edge_weight=edge_weight))
        return out

    def message(self, x_j, edge_attr, edge_weight=None):

        if edge_weight is not None:
            mess = F_act((x_j + edge_attr) * edge_weight)
        else:
            mess = F_act(x_j + edge_attr)
        return mess

        # return F_act(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GINMolHeadEncoder(torch.nn.Module):
 
    def __init__(self, num_layer, emb_dim, drop_ratio=0.5, JK="last", residual=True):
        
        super(GINMolHeadEncoder, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.residual = residual
        self.atom_encoder = AtomEncoder(emb_dim)
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            self.convs.append(GINConv(emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, x, edge_index, edge_attr, batch):
        
        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F_act(h), self.drop_ratio, training = self.training)
            if self.residual:
                h = h + h_list[layer]
            h_list.append(h)

        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]
        return node_representation


class GraphMolMasker(torch.nn.Module):

    def __init__(self, num_layer, emb_dim):
        super(GraphMolMasker, self).__init__()

        self.gnn_encoder = GINMolHeadEncoder(num_layer, emb_dim)
        self.edge_att_mlp = nn.Linear(emb_dim * 2, 1)
        self.node_att_mlp = nn.Linear(emb_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gnn_encoder)
        reset(self.edge_att_mlp)
        reset(self.node_att_mlp)

    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        node_rep = self.gnn_encoder(x, edge_index, edge_attr, batch)
        size = batch[-1].item() + 1
        row, col = edge_index
        edge_rep = torch.cat([node_rep[row], node_rep[col]], dim=-1)
        
        # if self.gumbel:
        #     node_key = self.gumbel_sigmoid(self.node_att_mlp(node_rep), temp=self.temp).unsqueeze(1)
        #     edge_key = self.gumbel_sigmoid(self.edge_att_mlp(edge_rep), temp=self.temp).unsqueeze(1)
        # else:
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
    
    def gumbel_sigmoid(self, logits, temp=0.1, bias=1e-4):

        device = logits.device
        eps = (bias - (1 - bias)) * torch.rand(logits.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(device)
        gate_inputs = (gate_inputs + logits) / temp
        weight = torch.sigmoid(gate_inputs).squeeze()
        return weight

    def reg_mask(self, mask, batch, size):

        key_num = scatter_add(mask, batch, dim=0, dim_size=size)
        env_num = scatter_add((1 - mask), batch, dim=0, dim_size=size)
        non_zero_mask = scatter_add((mask > 0).to(torch.float32), batch, dim=0, dim_size=size) 
        all_mask = scatter_add(torch.ones_like(mask).to(torch.float32), batch, dim=0, dim_size=size)
        non_zero_ratio = non_zero_mask / (all_mask + 1e-8)
        return key_num + 1e-8, env_num + 1e-8, non_zero_ratio



class GNNMolTailEncoder(torch.nn.Module):
 
    def __init__(self, num_layer, emb_dim, drop_ratio=0.5, JK="last", residual=True):
        
        super(GNNMolTailEncoder, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.residual = residual
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            self.convs.append(GINConv(emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, x, edge_index, edge_attr, batch, node_adv=None, edge_adv=None):
        
        if node_adv is not None:
            x = x * node_adv
        h_list = [x]
        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr, edge_adv)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F_act(h), self.drop_ratio, training = self.training)
            if self.residual:
                h = h + h_list[layer]
            h_list.append(h)

        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]
        return node_representation


# class GNNMolTailEncoder(torch.nn.Module):
#     def __init__(self, num_layer, emb_dim):
       
#         super(GNNMolTailEncoder, self).__init__()
#         self.num_layer = num_layer
#         self.emb_dim = emb_dim
#         self.dropout_rate = 0.5
#         self.relu1 = nn.ReLU()
#         self.relus = nn.ModuleList([nn.ReLU() for _ in range(num_layer - 1)])
#         self.batch_norm1 = nn.BatchNorm1d(emb_dim)
#         self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layer - 1)])
#         self.dropout1 = nn.Dropout(self.dropout_rate)
#         self.dropouts = nn.ModuleList([nn.Dropout(self.dropout_rate) for _ in range(num_layer - 1)])
#         self.conv1 = GINEConv(emb_dim)
#         self.convs = nn.ModuleList([GINEConv(emb_dim) for _ in range(num_layer - 1)])

#     def forward(self, x, edge_index, edge_attr, batch, node_adv=None, edge_adv=None):
        
#         if node_adv is not None:
#             x = x * node_adv
#         post_conv = self.batch_norm1(self.conv1(x, edge_index, edge_attr, edge_adv))
#         if self.num_layer > 1:
#             post_conv = self.relu1(post_conv)
#             post_conv = self.dropout1(post_conv)
        
#         for i, (conv, batch_norm, relu, dropout) in enumerate(zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
#             post_conv = batch_norm(conv(post_conv, edge_index, edge_attr, edge_adv))
#             if i != len(self.convs) - 1:
#                 post_conv = relu(post_conv)
#             post_conv = dropout(post_conv)
#         return post_conv





