import torch
import numpy as np
import random
import pdb
from torch_geometric.transforms import BaseTransform

def get_info_dataset(args, dataset, split_idx):

    total = []
    for mode in ['train', 'valid', 'test']:
        mode_max_node = 0
        mode_min_node = 9999
        mode_avg_node = 0
        mode_tot_node = 0.0

        dataset_name = dataset[split_idx[mode]]
        mode_num_graphs = len(dataset_name)
        for data in dataset_name:
            num_node = data.num_nodes
            mode_tot_node += num_node
            if num_node > mode_max_node:
                mode_max_node = num_node
            if num_node < mode_min_node:
                mode_min_node = num_node
        print("{} {:<5} | Graphs num:{:<5} | Node num max:{:<4}, min:{:<4}, avg:{:.2f}"
            .format(args.dataset, mode, mode_num_graphs,
                                        mode_max_node,
                                        mode_min_node, 
                                        mode_tot_node / mode_num_graphs))
        total.append(mode_num_graphs)
    all_graph_num = sum(total)
    print("train:{:.2f}%, val:{:.2f}%, test:{:.2f}%"
        .format(float(total[0]) * 100 / all_graph_num, 
                float(total[1]) * 100 / all_graph_num, 
                float(total[2]) * 100 / all_graph_num))

def size_split_idx(dataset, mode):

    num_graphs = len(dataset)
    num_val   = int(0.1 * num_graphs)
    num_test  = int(0.1 * num_graphs)
    num_train = num_graphs - num_test - num_val

    num_node_list = []
    train_idx = []
    valtest_list = []

    for data in dataset:
        num_node_list.append(data.num_nodes)

    sort_list = np.argsort(num_node_list)

    if mode == 'ls':
        train_idx = sort_list[2 * num_val:]
        valid_test_idx = sort_list[:2 * num_val]
    else:
        train_idx = sort_list[:-2 * num_val]
        valid_test_idx = sort_list[-2 * num_val:]
    random.shuffle(valid_test_idx)
    valid_idx = valid_test_idx[:num_val]
    test_idx = valid_test_idx[num_val:]

    split_idx = {'train': torch.tensor(train_idx, dtype = torch.long), 
                 'valid': torch.tensor(valid_idx, dtype = torch.long), 
                 'test': torch.tensor(test_idx, dtype = torch.long)}
    return split_idx
    
 

class ToEnvs(BaseTransform):
    
    def __init__(self, envs=10):
        self.envs = envs

    def __call__(self, data):

        data.env_id = torch.randint(0, self.envs, (1,))
        return data
