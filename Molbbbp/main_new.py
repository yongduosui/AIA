from GOOD import config_summoner
from GOOD.utils.args import args_parser
from GOOD.utils.logger import load_logger
from GOOD.kernel.pipeline import initialize_model_dataset
from GOOD.ood_algorithms.ood_manager import load_ood_alg
# from GOOD.kernel.pipeline import load_task
# from GOOD.kernel.train import train
import numpy as np
import pdb
from GOOD.networks.model_manager import config_model
from GOOD.kernel.evaluation import evaluate
from GOOD.kernel.train import train_batch
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from utils import *
from torch_geometric.loader import DataLoader
from GOOD.utils.initial import init
from GOOD.networks.model_manager import load_model
from GOOD.data import load_dataset, create_dataloader
from munch import Munch
from GOOD.data.dataset_manager import read_meta_info
from utils import ToEnvs


def run(config, args):
    
    # model, loader_orig = initialize_model_dataset(config)
    init(config)
    # dataset = load_dataset(config.dataset.dataset_name, config)
    dataset = PygGraphPropPredDataset(name=args.dataset_mol, transform=ToEnvs(10))
    num_graphs = len(dataset)
    # dataset.data.env_id = torch.randint(0, 10, (num_graphs,))
    meta_info = Munch()
    meta_info.dataset_type = 'mol'
    meta_info.model_level = 'graph'
    meta_info.dim_node = dataset.num_node_features
    meta_info.dim_edge = dataset.num_edge_features
    meta_info.num_envs = 10
    meta_info.num_classes = dataset.data.y.shape[1]

    read_meta_info(meta_info, config)

    config.metric.set_score_func('ROC-AUC')
    config.metric.set_loss_func('Binary classification')

    model = load_model(config.model.model_name, config)
    # print(model)
    ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
    
    if args.domain_mol == "scaffold":
        split_idx = dataset.get_idx_split()
    else:
        split_idx = size_split_idx(dataset, args.size)
    print("[!] split:{}, size:{}".format(args.domain_mol, args.size))
    get_info_dataset(args, dataset, split_idx)

    test_batch_size = 128
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,  num_workers=0)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=test_batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(dataset[split_idx["test"]],  batch_size=test_batch_size, shuffle=False, num_workers=0)
    print("bs:{}".format(args.batch_size))
    # for data in train_loader:
    #     pdb.set_trace()
    # pdb.set_trace()
    loader = {"train": train_loader, "val": valid_loader, "test": test_loader}
    config_model(model, 'train', config)

    config.train_helper.set_up(model, config)

    results = {'highest_valid_ood': 0, 
               'highest_valid_iid': 0,
               'update_test_ood': 0,  
               'update_test_iid': 0, 
               'update_epoch_ood': 0, 
               'update_epoch_iid': 0}

    train_name = config.ood.ood_alg
    for epoch in range(1, args.epochs + 1):
        mean_loss = 0
        spec_loss = 0
        total = len(loader['train'])
        show  = int(float(total) / 5.0)
        for index, data in enumerate(loader['train']):
            if data.batch is not None and (data.batch[-1] < config.train.train_bs - 1):
                continue
            p = (index / len(loader['train']) + epoch) / args.epochs
            config.train.alpha = 2. / (1. + np.exp(-10 * p)) - 1
            train_stat = train_batch(model, data, ood_algorithm, None, config)
            mean_loss = (mean_loss * index + ood_algorithm.mean_loss) / (index + 1)
            if config.ood.ood_alg not in ['ERM', 'GroupDRO', 'Mixup']:
                spec_loss = (spec_loss * index + ood_algorithm.spec_loss) / (index + 1)
                if index % show == 0:
                    print("{}, epoch:{}/{}, it:{:<4}/{:<4}, mean loss:{:.4f}, spec loss:{:.4f}"
                    .format(train_name, epoch, args.epochs, index, total, mean_loss, spec_loss))
            else:
                if index % show == 0:
                    print("{}, epoch:{}/{}, it:{:<4}/{:<4}, mean loss:{:.4f}"
                    .format(train_name, epoch, args.epochs, index, total, mean_loss))
        # train_acc = evaluate(model, loader, ood_algorithm, 'train', config)['score']
        valid_result_ood = evaluate(model, loader, ood_algorithm, 'val', config)['score']
        test_result_ood = evaluate(model, loader, ood_algorithm, 'test', config)['score']
        config.train_helper.scheduler.step()

        if valid_result_ood > results['highest_valid_ood']:
            results['highest_valid_ood'] = valid_result_ood
            results['update_test_ood'] = test_result_ood
            results['update_epoch_ood'] = epoch

        print("-"*150)
        print("{}, Epoch:[{:<3}/{:<3}], loss:[{:.4f}], valid:[{:.2f}], test:[{:.2f}] | Best val:[{:.2f}] Update test:[{:.2f}]"
            .format(train_name,
                    epoch, args.epochs, 
                    mean_loss,
                    valid_result_ood * 100,
                    test_result_ood * 100,
                    results['highest_valid_ood'] * 100,
                    results['update_test_ood'] * 100,
                    results['update_epoch_ood']))
        print("-"*150)
    # print("syd: Update test:[{:.2f}] at epoch:[{}]".format(results['update_test_ood'] * 100, results['update_epoch_ood']))
    return results['update_test_ood'] * 100

def main():

    ood_list = []
    run_time = 10
    args = args_parser()
    config = config_summoner(args)
    for i in range(run_time):
        test_ood = run(config, args)
        print("syd: run:{}/{}, test ood:{:.2f}".format(i + 1, run_time, test_ood))
        ood_list.append(test_ood)

    print('sydfinall: alg:{}, domain:{}, size:{}, Test OOD:{:.2f}±{:.2f}'
        .format(config.ood.ood_alg, args.domain_mol, args.size, np.mean(ood_list), np.std(ood_list)))
main()