import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
# from conv_syn import GNN_node, GNN_node_Virtualnode, GNNSynEncoder
# from conv_syn import GNNSynEncoder
from torch_scatter import scatter_mean
from torch_geometric.nn import Sequential, GINConv
import pdb


class GINNet(torch.nn.Module):
    
    def __init__(self, first_dim, num_classes, num_layer=5, emb_dim=300, dropout_rate=0.5):
        super(GINNet, self).__init__()

        self.first_dim = first_dim
        self.num_classes = num_classes
        self.num_layer = num_layer
        self.emb_dim = emb_dim

        self.virtual_node_embedding = nn.Embedding(1, emb_dim)
        self.virtual_mlp = nn.Sequential(*(
                [nn.Linear(emb_dim, 2 * emb_dim),
                 nn.BatchNorm1d(2 * emb_dim), nn.ReLU()] +
                [nn.Linear(2 * emb_dim, emb_dim),
                 nn.BatchNorm1d(emb_dim), nn.ReLU(),
                 nn.Dropout(dropout_rate)]))
        self.virtual_pool = global_add_pool
        
        self.conv1 = GINConv(nn.Sequential(nn.Linear(first_dim, 2 * emb_dim),
                                           nn.BatchNorm1d(2 * emb_dim), 
                                           nn.ReLU(),
                                           nn.Linear(2 * emb_dim, emb_dim)))

        self.convs = nn.ModuleList([GINConv(nn.Sequential(nn.Linear(emb_dim, 2 * emb_dim),
                                                          nn.BatchNorm1d(2 * emb_dim), 
                                                          nn.ReLU(),
                                                          nn.Linear(2 * emb_dim, emb_dim))) for _ in range(num_layer - 1)])
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        self.batch_norm1 = nn.BatchNorm1d(emb_dim)
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(emb_dim)
            for _ in range(num_layer - 1)
        ])
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout_rate)
            for _ in range(num_layer - 1)
        ])
        self.readout = global_mean_pool
        self.graph_pred_linear = nn.Linear(emb_dim, num_classes)
        
        
    def forward(self, batched_data):

        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        virtual_node_feat = self.virtual_node_embedding(
            torch.zeros(batch[-1].item() + 1, device=edge_index.device, dtype=torch.long))
        # virtual_node_feat = self.virtual_node_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            # --- Add global info ---
            post_conv = post_conv + virtual_node_feat[batch]
            post_conv = batch_norm(conv(post_conv, edge_index))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)
            # --- update global info ---
            if i < len(self.convs) - 1:
                virtual_node_feat = self.virtual_mlp(self.virtual_pool(post_conv, batch) + virtual_node_feat)

        out_readout = self.readout(post_conv, batch)
        pred = self.graph_pred_linear(out_readout)
        return pred
