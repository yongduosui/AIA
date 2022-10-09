r"""Training pipeline: training/evaluation structure, batch training.
"""

from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data.batch import Batch
# from tqdm import tqdm

from GOOD.kernel.evaluation import evaluate
from GOOD.networks.model_manager import config_model
from GOOD.ood_algorithms.algorithms.BaseOOD import BaseOODAlg
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from GOOD.utils.logger import pbar_setting
from GOOD.utils.train import nan2zero_get_mask
import pdb

def train_batch(model: torch.nn.Module, data: Batch, ood_algorithm: BaseOODAlg, pbar,
                config: Union[CommonArgs, Munch]) -> dict:
    r"""
    Train a batch. (Project use only)

    Args:
        model (torch.nn.Module): The GNN model.
        data (Batch): Current batch of data.
        ood_algorithm (BaseOODAlg: The OOD algorithm.
        config (Union[CommonArgs, Munch]): Please refer to :ref:`configs:GOOD Configs and command line Arguments (CA)`.

    Returns:
        Calculated loss.
    """
    data = data.to(config.device)

    config.train_helper.optimizer.zero_grad()

    mask, targets = nan2zero_get_mask(data, 'train', config)
    node_norm = data.node_norm if config.model.model_level == 'node' else None
    data, targets, mask, node_norm = ood_algorithm.input_preprocess(data, targets, mask, node_norm, model.training,
                                                                    config)
    edge_weight = data.edge_norm if config.model.model_level == 'node' else None

    model_output = model(data=data, edge_weight=edge_weight, ood_algorithm=ood_algorithm)
    raw_pred = ood_algorithm.output_postprocess(model_output)

    loss = ood_algorithm.loss_calculate(raw_pred, targets, mask, node_norm, config)
    loss = ood_algorithm.loss_postprocess(loss, data, mask, config)
    loss.backward()

    config.train_helper.optimizer.step()

    return {'loss': loss.detach()}


def train(model: torch.nn.Module, loader: Union[DataLoader, Dict[str, DataLoader]], ood_algorithm: BaseOODAlg,
          config: Union[CommonArgs, Munch]):
    
    # print('#D#Config model')
    config_model(model, 'train', config)

    # print('#D#Load training utils')
    config.train_helper.set_up(model, config)

    best_vaild_iid, best_vaild_ood, update_test_iid, update_test_ood = 0, 0, 0, 0
    train_name = config.ood.ood_alg
    for epoch in range(config.train.ctn_epoch, config.train.max_epoch):
        mean_loss = 0
        spec_loss = 0
        total = len(loader['train'])
        show  = int(float(total) / 5.0)
        for index, data in enumerate(loader['train']):
            if data.batch is not None and (data.batch[-1] < config.train.train_bs - 1):
                continue
            p = (index / len(loader['train']) + epoch) / config.train.max_epoch
            config.train.alpha = 2. / (1. + np.exp(-10 * p)) - 1
            train_stat = train_batch(model, data, ood_algorithm, None, config)
            mean_loss = (mean_loss * index + ood_algorithm.mean_loss) / (index + 1)
            if config.ood.ood_alg not in ['ERM', 'GroupDRO', 'Mixup']:
                spec_loss = (spec_loss * index + ood_algorithm.spec_loss) / (index + 1)
                if index % show == 0:
                    print("{}, epoch:{}/{}, it:{}/{}, mean loss:{:.4f}, spec loss:{:.4f}"
                    .format(train_name, epoch + 1, config.train.max_epoch, index, total, mean_loss, spec_loss))
            else:
                if index % show == 0:
                    print("{}, epoch:{}/{}, it:{}/{}, mean loss:{:.4f}"
                    .format(train_name, epoch + 1, config.train.max_epoch, index, total, mean_loss))

        train_acc = evaluate(model, loader, ood_algorithm, 'eval_train', config)['score']
        val_iid_acc = evaluate(model, loader, ood_algorithm, 'id_val', config)['score']
        test_iid_acc = evaluate(model, loader, ood_algorithm, 'id_test', config)['score']
        val_ood_acc = evaluate(model, loader, ood_algorithm, 'val', config)['score']
        test_ood_acc = evaluate(model, loader, ood_algorithm, 'test', config)['score']
        config.train_helper.scheduler.step()

        if best_vaild_iid < val_iid_acc:
            best_vaild_iid = val_iid_acc
            update_test_iid = test_iid_acc
        if best_vaild_ood < val_ood_acc:
            best_vaild_ood = val_ood_acc
            update_test_ood = test_ood_acc
        print("-"*150)
        print("{}, epoch:{}/{}, train:{:.2f}, val:{:.2f}/{:.2f}, test:{:.2f}/{:.2f} | best val:{:.2f}/{:.2f} update test:{:.2f}/{:.2f}"
            .format(train_name,
                    epoch + 1, 
                    config.train.max_epoch, 
                    train_acc * 100,
                    val_iid_acc * 100,
                    val_ood_acc * 100,
                    test_iid_acc * 100,
                    test_ood_acc * 100,
                    best_vaild_iid * 100,
                    best_vaild_ood * 100,
                    update_test_iid * 100,
                    update_test_ood * 100))
        print("-"*150)

    return update_test_iid * 100, update_test_ood * 100


