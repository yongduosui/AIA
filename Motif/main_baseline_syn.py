import torch
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import torch.optim as optim
import torch.nn.functional as F
from models import CausalAdvGNN
import argparse
import time
import numpy as np
from GOOD.data.good_datasets.good_cmnist import GOODCMNIST
from GOOD.data.good_datasets.good_motif import GOODMotif
from gnn import GINNet
import pdb

def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        print(arg + '.' * (str_num - len(arg) - len(str(val))) + str(val))
    print()

def eval(model, loader, device):
    
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)

def main(args):
    
    num_classes = 10
    first_dim = 3
    eval_metric = "rocauc"
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    if args.dataset == "cmnist":
        dataset, meta_info = GOODCMNIST.load("dataset", domain=args.domain, shift='covariate', generate=False)
    elif args.dataset == "motif":
        dataset, meta_info = GOODMotif.load("dataset", domain=args.domain, shift='covariate', generate=False)
        num_class = 3
        num_layer = 3
        in_dim = 1
    else:
        assert False
    train_loader = DataLoader(dataset["train"], batch_size=args.batch_size, shuffle=True)
    valid_loader_ood = DataLoader(dataset["val"], batch_size=args.batch_size, shuffle=False)
    valid_loader_iid = DataLoader(dataset["id_val"], batch_size=args.batch_size, shuffle=False)
    test_loader_ood = DataLoader(dataset["test"],  batch_size=args.batch_size, shuffle=False)
    test_loader_iid  = DataLoader(dataset["id_test"],  batch_size=args.batch_size, shuffle=False)
    model = GINNet(num_class, dataset, num_layer, in_dim, emb_dim=300).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    results = {'highest_valid_ood': 0, 
               'highest_valid_iid': 0,
               'update_test_ood': 0,  
               'update_test_iid': 0, 
               'update_epoch_ood': 0, 
               'update_epoch_iid': 0}
    start_time = time.time()

    for epoch in range(args.epochs):
        start_time_local = time.time()
        total_loss = 0
        model.train()
        show  = int(float(len(train_loader)) / 5.0)
        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            pred = model(batch)
            optimizer.zero_grad()
            loss = criterion(pred, batch.y.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if step % show == 0:
                print("Epoch:[{}/{}] Train Iter:[{:<3}/{}] Loss:[{:.4f}]".format(epoch, args.epochs, step, len(train_loader), total_loss / (step + 1)))
        epoch_loss = total_loss / len(train_loader)
        valid_result_iid = eval(model, valid_loader_iid, device)
        valid_result_ood = eval(model, valid_loader_ood, device)
        test_result_iid = eval(model, test_loader_iid, device)
        test_result_ood = eval(model, test_loader_ood, device)
    
        if valid_result_iid > results['highest_valid_iid']:
            results['highest_valid_iid'] = valid_result_iid
            results['update_test_iid'] = test_result_iid
            results['update_epoch_iid'] = epoch
        if valid_result_ood > results['highest_valid_ood']:
            results['highest_valid_ood'] = valid_result_ood
            results['update_test_ood'] = test_result_ood
            results['update_epoch_ood'] = epoch
        print("-" * 150)
        print("Epoch:[{}/{}], loss:[{:.4f}], valid:[{:.2f}/{:.2f}], test:[{:.2f}/{:.2f}] | Best val:[{:.2f}/{:.2f}] Update test:[{:.2f}/{:.2f}] at:[{}/{}] | epoch time:{:.2f} min"
                        .format(epoch, 
                                args.epochs, 
                                epoch_loss, 
                                valid_result_iid * 100,
                                valid_result_ood * 100,
                                test_result_iid * 100,
                                test_result_ood * 100,
                                results['highest_valid_iid'] * 100,
                                results['highest_valid_ood'] * 100,
                                results['update_test_iid'] * 100,
                                results['update_test_ood'] * 100,
                                results['update_epoch_iid'],
                                results['update_epoch_ood'],
                                (time.time()-start_time_local) / 60))
        print("-" * 150)
    total_time = time.time() - start_time
    print("syd: Update test:[{:.2f}/{:.2f}] at epoch:[{}/{}] | Total time:{}"
            .format(results['update_test_iid'] * 100,
                    results['update_test_ood'] * 100,
                    results['update_epoch_iid'],
                    results['update_epoch_ood'],
                    time.strftime('%H:%M:%S', time.gmtime(total_time))))
    return results['update_test_iid'], results['update_test_ood']

def config_and_run(args):
    
    print_args(args)
    final_test_acc_iid = []
    final_test_acc_ood = []
    for _ in range(args.trails):
        test_auc_iid, test_auc_ood = main(args)
        final_test_acc_iid.append(test_auc_iid)
        final_test_acc_ood.append(test_auc_ood)
    print("sydfinall: Test ACC IID: [{:.2f}±{:.2f}], OOD: [{:.2f}±{:.2f}]"
        .format(np.mean(final_test_acc_iid) * 100, np.std(final_test_acc_iid) * 100, 
                np.mean(final_test_acc_ood) * 100, np.std(final_test_acc_ood) * 100))
    print("all IID:{}, all OOD:{}".format(final_test_acc_iid, final_test_acc_ood))
if __name__ == "__main__":

    def arg_parse():
        str2bool = lambda x: x.lower() == "true"
        parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
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
