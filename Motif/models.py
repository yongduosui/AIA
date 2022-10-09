import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from conv_syn import GINConv, GNNSynEncoder, GraphSynMasker
from conv_mol import GINMolHeadEncoder, GraphMolMasker, GNNMolTailEncoder, vGINMolHeadEncoder
import pdb

class CausalAdvGNNSyn(torch.nn.Module):

    def __init__(self, num_class, 
                       in_dim,
                       emb_dim=300,
                       fro_layer=2,
                       bac_layer=2,
                       cau_layer=2,
                       att_layer=2,
                       dropout_rate=0.5,
                       cau_gamma=0.4,
                       adv_gamma=1.0,
                       adv_gamma_edge=1.0):

        super(CausalAdvGNNSyn, self).__init__()

        self.cau_gamma = cau_gamma
        self.adv_gamma = adv_gamma
        self.adv_gamma_edge = adv_gamma_edge
        self.dropout_rate = dropout_rate
        self.emb_dim = emb_dim
        self.num_class = num_class
        self.wasserstein_distance = nn.MSELoss()
        self.graph_front = GNNSynEncoder(fro_layer, in_dim,  emb_dim, dropout_rate)
        self.graph_backs = GNNSynEncoder(bac_layer, emb_dim, emb_dim, dropout_rate)
        self.causaler = GraphSynMasker(cau_layer, in_dim, emb_dim, dropout_rate)
        self.attacker = GraphSynMasker(att_layer, in_dim, emb_dim, dropout_rate)
        self.pool = global_mean_pool
        self.predictor = torch.nn.Linear(emb_dim, num_class)
        
    def forward_causal(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_encode = self.graph_front(x, edge_index)
        causaler_output = self.causaler(data)
        node_cau, edge_cau = causaler_output["node_key"], causaler_output["edge_key"]
        h_node_cau = self.graph_backs(x_encode, edge_index, node_cau, edge_cau)
        h_graph_cau = self.pool(h_node_cau, batch)
        pred_cau = self.predictor(h_graph_cau)
        return pred_cau

    def forward_combined_inference(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_encode = self.graph_front(x, edge_index)
        attacker_output = self.attacker(data)
        causaler_output = self.causaler(data)
        node_adv, edge_adv = attacker_output["node_key"], attacker_output["edge_key"]
        node_cau, edge_cau = causaler_output["node_key"], causaler_output["edge_key"]
        node_com = (1 - node_cau) * node_adv + node_cau
        edge_com = (1 - edge_cau) * edge_adv + edge_cau
        h_node_com = self.graph_backs(x_encode, edge_index, node_com, edge_com)
        h_graph_com = self.pool(h_node_com, batch)
        pred_com = self.predictor(h_graph_com)
        return pred_com

    def forward_attack(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_encode = self.graph_front(x, edge_index)

        attacker_output = self.attacker(data)
        node_adv, edge_adv = attacker_output["node_key"], attacker_output["edge_key"]
        node_adv_num, node_env_num = attacker_output["node_key_num"], attacker_output["node_env_num"]
        edge_adv_num, edge_env_num = attacker_output["edge_key_num"], attacker_output["edge_env_num"]

        h_node_adv = self.graph_backs(x_encode, edge_index, node_adv, edge_adv)
        h_node_ori = self.graph_backs(x_encode, edge_index)
        h_graph_adv = self.pool(h_node_adv, batch)
        h_graph_ori = self.pool(h_node_ori, batch)

        loss_dis = self.wasserstein_distance(h_graph_adv, h_graph_ori)
        pred_adv = self.predictor(h_graph_adv)

        adv_node_reg = self.reg_mask_loss(node_adv_num, node_env_num, self.adv_gamma, self.attacker.non_zero_node_ratio)
        adv_edge_reg = self.reg_mask_loss(edge_adv_num, edge_env_num, self.adv_gamma * self.adv_gamma_edge, self.attacker.non_zero_edge_ratio)
        adv_loss_reg = adv_node_reg + adv_edge_reg

        output = {'pred_adv': pred_adv, 'loss_dis': loss_dis,
                  'adv_loss_reg': adv_loss_reg,
                  'node_adv': node_adv.mean().item(),
                  'edge_adv': edge_adv.mean().item()}
        return output

    def forward_advcausal(self, data, vis=False):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_encode = self.graph_front(x, edge_index)
        attacker_output = self.attacker(data)
        causaler_output = self.causaler(data)
        node_adv, edge_adv = attacker_output["node_key"], attacker_output["edge_key"]
        node_cau, edge_cau = causaler_output["node_key"], causaler_output["edge_key"]
        node_cau_num, node_env_num = causaler_output["node_key_num"], causaler_output["node_env_num"]
        edge_cau_num, edge_env_num = causaler_output["edge_key_num"], causaler_output["edge_env_num"]

        node_com = (1 - node_cau) * node_adv + node_cau
        edge_com = (1 - edge_cau) * edge_adv + edge_cau
        
        h_node_cau = self.graph_backs(x_encode, edge_index, node_cau, edge_cau)
        h_graph_cau = self.pool(h_node_cau, batch)
        pred_cau = self.predictor(h_graph_cau)

        h_node_com = self.graph_backs(x_encode, edge_index, node_com, edge_com)
        h_graph_com = self.pool(h_node_com, batch)
        pred_com = self.predictor(h_graph_com)
        
        cau_node_reg = self.reg_mask_loss(node_cau_num, node_env_num, self.cau_gamma, self.causaler.non_zero_node_ratio)
        cau_edge_reg = self.reg_mask_loss(edge_cau_num, edge_env_num, self.cau_gamma, self.causaler.non_zero_edge_ratio)
        cau_loss_reg = cau_node_reg + cau_edge_reg
        
        if vis:
            self.plot_state(node_cau, edge_cau, "cau")
            self.plot_state(node_adv, edge_adv, "adv")
            self.plot_state(node_com, edge_com, "com")

            return node_cau.cpu(), edge_cau.cpu(), node_adv.cpu(), edge_adv.cpu(), node_com.cpu(), edge_com.cpu()
        pred_advcausal = {'pred_cau': pred_cau, 'pred_com': pred_com, 
                  'cau_loss_reg':cau_loss_reg,
                  'node_cau': node_cau.mean().item(),
                  'edge_cau': edge_cau.mean().item(),
                  'node_adv': node_adv.mean().item(),
                  'edge_adv': edge_adv.mean().item(),
                  'node_com': node_com.mean().item(),
                  'edge_com': edge_com.mean().item()}

        return pred_advcausal

    def reg_mask_loss(self, key_mask, env_mask, gamma, non_zero_ratio):

        loss_reg =  torch.abs(key_mask / (key_mask + env_mask) - gamma * torch.ones_like(key_mask)).mean()
        loss_reg += (non_zero_ratio - gamma  * torch.ones_like(key_mask)).mean()
        return loss_reg

    def plot_state(self, node, edge, name):
        print("{} | node mean:{:.4f}, max:{:.4f}, min:{:.4f}, edge mean:{:.4f}, max:{:.4f}, min:{:.4f}"
                .format(name, node.mean(), node.max(), node.min(),
                        edge.mean(), edge.max(), edge.min()))

class CausalAdvGNNMol(torch.nn.Module):

    def __init__(self, num_class, 
                       emb_dim=300,
                       fro_layer=2,
                       bac_layer=2,
                       cau_layer=2,
                       att_layer=2,
                       cau_gamma=0.4,
                       adv_gamma=1.0):

        super(CausalAdvGNNMol, self).__init__()

        self.cau_gamma = cau_gamma
        self.adv_gamma = adv_gamma
        self.dropout_rate = 0.5
        self.emb_dim = emb_dim
        self.num_class = num_class
        self.wasserstein_distance = nn.MSELoss()
        # self.graph_front = GINMolHeadEncoder(fro_layer, emb_dim)
        self.graph_front = vGINMolHeadEncoder(fro_layer, emb_dim)
        self.graph_backs = GNNMolTailEncoder(bac_layer, emb_dim)
        self.causaler = GraphMolMasker(cau_layer, emb_dim)
        self.attacker = GraphMolMasker(att_layer, emb_dim)
        self.pool = global_mean_pool
        self.predictor = torch.nn.Linear(emb_dim, num_class)
        
    def forward_causal(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x_encode = self.graph_front(x, edge_index, edge_attr, batch)
        causaler_output = self.causaler(data)
        node_cau, edge_cau = causaler_output["node_key"], causaler_output["edge_key"]
        h_node_cau = self.graph_backs(x_encode, edge_index, edge_attr, node_cau, edge_cau)
        h_graph_cau = self.pool(h_node_cau, batch)
        pred_cau = self.predictor(h_graph_cau)
        return pred_cau

    def forward_attack(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x_encode = self.graph_front(x, edge_index, edge_attr, batch)

        attacker_output = self.attacker(data)
        node_adv, edge_adv = attacker_output["node_key"], attacker_output["edge_key"]
        node_adv_num, node_env_num = attacker_output["node_key_num"], attacker_output["node_env_num"]
        edge_adv_num, edge_env_num = attacker_output["edge_key_num"], attacker_output["edge_env_num"]

        h_node_adv = self.graph_backs(x_encode, edge_index, edge_attr, node_adv, edge_adv)
        h_node_ori = self.graph_backs(x_encode, edge_index, edge_attr)
        h_graph_adv = self.pool(h_node_adv, batch)
        h_graph_ori = self.pool(h_node_ori, batch)

        loss_dis = self.wasserstein_distance(h_graph_adv, h_graph_ori)
        pred_adv = self.predictor(h_graph_adv)

        adv_node_reg = self.reg_mask_loss(node_adv_num, node_env_num, self.adv_gamma, self.attacker.non_zero_node_ratio)
        adv_edge_reg = self.reg_mask_loss(edge_adv_num, edge_env_num, self.adv_gamma, self.attacker.non_zero_edge_ratio)
        adv_loss_reg = adv_node_reg + adv_edge_reg

        output = {'pred_adv': pred_adv, 'loss_dis': loss_dis,
                  'adv_loss_reg': adv_loss_reg,
                  'node_adv': node_adv.mean().item(),
                  'edge_adv': edge_adv.mean().item()}
        return output

    def forward_advcausal(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x_encode = self.graph_front(x, edge_index, edge_attr, batch)
        attacker_output = self.attacker(data)
        causaler_output = self.causaler(data)
        node_adv, edge_adv = attacker_output["node_key"], attacker_output["edge_key"]
        node_cau, edge_cau = causaler_output["node_key"], causaler_output["edge_key"]
        node_cau_num, node_env_num = causaler_output["node_key_num"], causaler_output["node_env_num"]
        edge_cau_num, edge_env_num = causaler_output["edge_key_num"], causaler_output["edge_env_num"]

        node_com = (1 - node_cau) * node_adv + node_cau
        edge_com = (1 - edge_cau) * edge_adv + edge_cau
        
        h_node_cau = self.graph_backs(x_encode, edge_index, edge_attr, node_cau, edge_cau)
        h_graph_cau = self.pool(h_node_cau, batch)
        pred_cau = self.predictor(h_graph_cau)

        h_node_com = self.graph_backs(x_encode, edge_index, edge_attr, node_com, edge_com)
        h_graph_com = self.pool(h_node_com, batch)
        pred_com = self.predictor(h_graph_com)
        
        cau_node_reg = self.reg_mask_loss(node_cau_num, node_env_num, self.cau_gamma, self.causaler.non_zero_node_ratio)
        cau_edge_reg = self.reg_mask_loss(edge_cau_num, edge_env_num, self.cau_gamma, self.causaler.non_zero_edge_ratio)
        cau_loss_reg = cau_node_reg + cau_edge_reg
        
        pred_advcausal = {'pred_cau': pred_cau, 'pred_com': pred_com, 
                  'cau_loss_reg':cau_loss_reg,
                  'node_cau': node_cau.mean().item(),
                  'edge_cau': edge_cau.mean().item(),
                  'node_adv': node_adv.mean().item(),
                  'edge_adv': edge_adv.mean().item(),
                  'node_com': node_com.mean().item(),
                  'edge_com': edge_com.mean().item()}

        return pred_advcausal

    def reg_mask_loss(self, key_mask, env_mask, gamma, non_zero_ratio):

        loss_reg =  torch.abs(key_mask / (key_mask + env_mask) - gamma * torch.ones_like(key_mask)).mean()
        loss_reg += (non_zero_ratio - gamma  * torch.ones_like(key_mask)).mean()
        return loss_reg

