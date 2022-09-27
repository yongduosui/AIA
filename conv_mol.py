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

class GINMolHeadEncoder(torch.nn.Module):

    def __init__(self, num_layer, emb_dim):
        super(GINMolHeadEncoder, self).__init__()
        
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.atom_encoder = AtomEncoder(emb_dim)
        self.conv1 = GINEConv(emb_dim)
        self.convs = nn.ModuleList([GINEConv(emb_dim) for _ in range(num_layer - 1)])
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [nn.ReLU() for _ in range(num_layer - 1)]
        )
        self.batch_norm1 = nn.BatchNorm1d(emb_dim)
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(emb_dim)
            for _ in range(num_layer - 1)
        ])
        self.dropout1 = nn.Dropout()
        self.dropouts = nn.ModuleList([
            nn.Dropout() for _ in range(num_layer - 1)
        ])

    def forward(self, x, edge_index, edge_attr):
        
        x = self.atom_encoder(x)
        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index, edge_attr))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = batch_norm(conv(post_conv, edge_index, edge_attr))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)
        return post_conv

class vGINMolHeadEncoder(torch.nn.Module):
    
    def __init__(self, num_layer, emb_dim):
        super(vGINMolHeadEncoder, self).__init__()

        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.atom_encoder = AtomEncoder(emb_dim)
        self.conv1 = GINEConv(emb_dim)
        self.convs = nn.ModuleList([GINEConv(emb_dim) for _ in range(num_layer - 1)])
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [nn.ReLU() for _ in range(num_layer - 1)]
        )
        self.batch_norm1 = nn.BatchNorm1d(emb_dim)
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(emb_dim)
            for _ in range(num_layer - 1)
        ])
        self.dropout1 = nn.Dropout()
        self.dropouts = nn.ModuleList([
            nn.Dropout() for _ in range(num_layer - 1)
        ])

        self.virtual_node_embedding = nn.Embedding(1, emb_dim)
        self.virtual_mlp = nn.Sequential(*(
                [nn.Linear(emb_dim, 2 * emb_dim),
                 nn.BatchNorm1d(2 * emb_dim), nn.ReLU()] +
                [nn.Linear(2 * emb_dim, emb_dim),
                 nn.BatchNorm1d(emb_dim), nn.ReLU(),
                 nn.Dropout()]
        ))
        self.virtual_pool = global_add_pool

    def forward(self, batched_data):

        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        virtual_node_feat = self.virtual_node_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
        x = self.atom_encoder(x)
        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index, edge_attr))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            # --- Add global info ---
            post_conv = post_conv + virtual_node_feat[batch]
            post_conv = batch_norm(conv(post_conv, edge_index, edge_attr))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)
            # --- update global info ---
            if i < len(self.convs) - 1:
                virtual_node_feat = self.virtual_mlp(self.virtual_pool(post_conv, batch) + virtual_node_feat)
        # out_readout = self.readout(post_conv, batch)
        return post_conv

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
        node_rep = self.gnn_encoder(x, edge_index, edge_attr)
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

class GNNMolTailEncoder(torch.nn.Module):
    def __init__(self, num_layer, emb_dim):
       
        super(GNNMolTailEncoder, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.dropout_rate = 0.5
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList([nn.ReLU() for _ in range(num_layer - 1)])
        self.batch_norm1 = nn.BatchNorm1d(emb_dim)
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layer - 1)])
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.dropouts = nn.ModuleList([nn.Dropout(self.dropout_rate) for _ in range(num_layer - 1)])
        self.conv1 = GINEConv(emb_dim)
        self.convs = nn.ModuleList([GINEConv(emb_dim) for _ in range(num_layer - 1)])

    def forward(self, x, edge_index, edge_attr, node_adv=None, edge_adv=None):
        
        if node_adv is not None:
            x = x * node_adv
        post_conv = self.batch_norm1(self.conv1(x, edge_index, edge_attr, edge_adv))
        if self.num_layer > 1:
            post_conv = self.relu1(post_conv)
            post_conv = self.dropout1(post_conv)
        
        for i, (conv, batch_norm, relu, dropout) in enumerate(zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = batch_norm(conv(post_conv, edge_index, edge_attr, edge_adv))
            if i != len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)
        return post_conv


class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        
        super(GINEConv, self).__init__(aggr = "add")
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), 
                                       torch.nn.BatchNorm1d(2*emb_dim), 
                                       torch.nn.ReLU(), 
                                       torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr, edge_weight=None):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding, edge_weight=edge_weight))
        return out

    def message(self, x_j, edge_attr, edge_weight=None):
        if edge_weight is not None:
            mess = F.relu((x_j + edge_attr) * edge_weight)
        else:
            mess = F.relu(x_j + edge_attr)
        return mess

    def update(self, aggr_out):
        return aggr_out


