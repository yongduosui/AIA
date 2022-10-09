import torch
from torch_geometric.loader import DataLoader
from ogb.graphproppred import Evaluator
import torch.optim as optim
import torch.nn.functional as F
import argparse
import time
import numpy as np
from GOOD.data.good_datasets.good_hiv import GOODHIV
from GOOD.data.good_datasets.good_pcba import GOODPCBA
from gnn import GINNet
import random
import os
import pdb

def set_seed(seed):

    np.random.seed(seed) 
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.enabled = False  
    torch.backends.cudnn.benchmark = False  

def _init_fn(worker_id): 
    random.seed(10 + worker_id)
    np.random.seed(10 + worker_id)
    torch.manual_seed(10 + worker_id)
    torch.cuda.manual_seed(10 + worker_id)
    torch.cuda.manual_seed_all(10 + worker_id)

def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        print(arg + '.' * (str_num - len(arg) - len(str(val))) + str(val))
    print()


def eval(model, evaluator, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    output = evaluator.eval(input_dict)
    return output


def main(args):
    
    set_seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if args.dataset == "hiv":
        dataset, meta_info = GOODHIV.load("dataset", domain=args.domain, shift='covariate', generate=False)
        num_class = 1
        num_layer = 3
        eval_metric = "rocauc"
        eval_name = "ogbg-molhiv"
        test_batch_size = 256
    elif args.dataset == "pcba":
        eval_name = "ogbg-molpcba"
        dataset, meta_info = GOODPCBA.load("dataset", domain=args.domain, shift='covariate', generate=False)
        num_class = 128
        num_layer = 5
        eval_metric = "ap"
        test_batch_size = 4096
    else:
        assert False
    evaluator = Evaluator(eval_name)
    train_loader     = DataLoader(dataset["train"],  batch_size=args.batch_size, shuffle=True,  num_workers=0, worker_init_fn=_init_fn)
    valid_loader_ood = DataLoader(dataset["val"],    batch_size=test_batch_size, shuffle=False, num_workers=0, worker_init_fn=_init_fn)
    test_loader_ood  = DataLoader(dataset["test"],   batch_size=test_batch_size, shuffle=False, num_workers=0, worker_init_fn=_init_fn)

    model = GINNet(num_class, args.dataset, num_layer, virtual=args.virtual, emb_dim=300).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    results = {'highest_valid': 0, 'update_test': 0,  'update_epoch': 0}

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        start_time_local = time.time()
        total_loss = 0
        model.train()
        show  = int(float(len(train_loader)) / 4.0)
        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            pred = model(batch)
            optimizer.zero_grad()
            is_labeled = batch.y == batch.y
            # loss = criterion(pred, batch.y.long())
            loss = criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if step % show == 0:
                print("Epoch:[{}/{}] Train Iter:[{:<3}/{}] Loss:[{:.4f}]".format(epoch, args.epochs, step, len(train_loader), total_loss / (step + 1)))
        epoch_loss = total_loss / len(train_loader)
        valid_result = eval(model, evaluator, valid_loader_ood, device)[eval_metric]
        test_result  = eval(model, evaluator, test_loader_ood,  device)[eval_metric]
    
        if valid_result > results['highest_valid']:
            results['highest_valid'] = valid_result
            results['update_test'] = test_result
            results['update_epoch'] = epoch

        print("-" * 150)
        print("Epoch:[{}/{}], loss:[{:.4f}], valid:[{:.2f}], test:[{:.2f}] | Best val:[{:.2f}] Update test:[{:.2f}] at:[{}] | epoch time:{:.2f} min"
                        .format(epoch, args.epochs, 
                                epoch_loss, 
                                valid_result * 100, 
                                test_result * 100,
                                results['highest_valid'] * 100,
                                results['update_test'] * 100,
                                results['update_epoch'],
                                (time.time()-start_time_local) / 60))
        print("-" * 150)
    total_time = time.time() - start_time
    print("syd: Update test:[{:.2f}] at epoch:[{}] | Total time:{}"
            .format(results['update_test'] * 100,
                    results['update_epoch'],
                    time.strftime('%H:%M:%S', time.gmtime(total_time))))
    return results['update_test']

def config_and_run(args):
    
    print_args(args)
    # set_seed(args.seed)
    final_test_acc = []
    for _ in range(args.trails):
        test_auc = main(args)
        final_test_acc.append(test_auc)
    print("sydfinall: Test ACC OOD: [{:.2f}±{:.2f}]".format(np.mean(final_test_acc) * 100, np.std(final_test_acc) * 100))
    print("ALL OOD:{}".format(final_test_acc))


if __name__ == "__main__":

    def arg_parse():
        str2bool = lambda x: x.lower() == "true"
        parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
        
        parser.add_argument('--virtual',type=str2bool,   default=False,   help='dropout ratio (default: 0.5)')
        parser.add_argument('--seed',   type=int,   default=123,   help='dropout ratio (default: 0.5)')
        parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
        parser.add_argument('--domain', type=str, default='color', help='color, background')
        parser.add_argument('--emb_dim', type=int, default=300)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--dataset', type=str, default="hiv")
        parser.add_argument('--trails', type=int, default=1, help='number of runs (default: 0)')   
        args = parser.parse_args()
        return args

    args = arg_parse()
    config_and_run(args)
