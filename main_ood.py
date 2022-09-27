import torch
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import torch.optim as optim
import torch.nn.functional as F
from models import CausalAdvGNN
import argparse
import time
import numpy as np
from GOOD.data.good_datasets.good_hiv import GOODHIV
from gnn import 
import pdb

def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        print(arg + '.' * (str_num - len(arg) - len(str(val))) + str(val))
    print()

cls_criterion = torch.nn.BCEWithLogitsLoss()
def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for name, param in net.named_parameters():
                param.requires_grad = requires_grad
                # print("name:{}, grad:{}".format(name, param.requires_grad))

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model.forward_causal(batch)
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)


def main(args):

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    dataset = PygGraphPropPredDataset(name = args.dataset)
    split_idx = dataset.get_idx_split()

    hiv_datasets, hiv_meta_info = GOODHIV.load(dataset_root, domain='scaffold', shift='covariate', generate=False)

    evaluator = Evaluator(args.dataset)
    train_loader = DataLoader(hiv_datasets["train"], batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    valid_loader = DataLoader(hiv_datasets["valid"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(hiv_datasets["test"],  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = CausalAdvGNN(gnn_type=args.gnn, 
                         num_tasks=dataset.num_tasks, 
                         fro_layer=args.fro_layer,
                         bac_layer=args.bac_layer,
                         cau_layer=args.cau_layer,
                         att_layer=args.att_layer, 
                         emb_dim=args.emb_dim,
                         drop_ratio=args.drop_ratio,
                         cau_gamma=args.cau_gamma,
                         adv_gamma=args.adv_gamma).to(device)

    opt_attacker = optim.Adam(model.attacker.parameters(), lr=args.attacker_lr, weight_decay=args.l2reg)
    opt_causaler = optim.Adam(list(model.graph_front.parameters()) 
                            + list(model.graph_backs.parameters()) 
                            + list(model.causaler.parameters())
                            + list(model.predictor.parameters()), 
                              lr=args.causaler_lr,
                              weight_decay=args.l2reg)

    optimizers = {'attacker': opt_attacker, 'causaler': opt_causaler}
    results = {'highest_valid': 0, 'update_test': 0, 'update_epoch': 0}
    start_time = time.time()

    path_list = [args.causaler_time, args.causaler_time + args.attacker_time]
    for epoch in range(args.epochs):
        start_time_local = time.time()

        path = epoch % int(path_list[-1])
        if path in list(range(int(path_list[0]))):
            optimizer_name = 'causaler'
        elif path in list(range(int(path_list[0]), int(path_list[1]))):
            optimizer_name = 'attacker'
        #################################### Begin training #####################################
        total_loss = 0
        total_loss_cau = 0
        total_loss_com = 0 
        total_loss_reg = 0
        total_loss_adv = 0
        total_loss_dis = 0

        node_cau_list, edge_cau_list, node_adv_list, edge_adv_list  = [], [], [], []
        model.train()
        optimizer = optimizers[optimizer_name]
        causaler_branch = [model.graph_front, model.graph_backs, model.causaler, model.predictor]
        attacker_branch = [model.attacker]
        if optimizer_name == 'causaler':
            set_requires_grad(causaler_branch, requires_grad=True)
            set_requires_grad(attacker_branch, requires_grad=False)
        elif optimizer_name == 'attacker':
            set_requires_grad(attacker_branch, requires_grad=True)
            set_requires_grad(causaler_branch, requires_grad=False)
        else:
            assert False

        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
                pass
            else:   
                is_labeled = batch.y == batch.y
                optimizer.zero_grad()
                if optimizer_name == 'causaler':
                    # if epoch > 10:
                    #     pdb.set_trace()
                    pred_advcausal = model.forward_advcausal(batch)
                    loss_cau = cls_criterion(pred_advcausal["pred_cau"].to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                    loss_com = cls_criterion(pred_advcausal["pred_com"].to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                    loss_reg = pred_advcausal["cau_loss_reg"]
                    node_cau_list.append(pred_advcausal["node_cau"])
                    edge_cau_list.append(pred_advcausal["edge_cau"])
                    node_adv_list.append(pred_advcausal["node_adv"])
                    edge_adv_list.append(pred_advcausal["edge_adv"])
                    loss = loss_cau + loss_com + loss_reg * args.cau_reg
                    loss.backward()
                    total_loss_cau += loss_cau.item()
                    total_loss_com += loss_com.item() 
                    total_loss_reg += loss_reg.item()
                else:
                    pred_attack = model.forward_attack(batch)
                    loss_adv = cls_criterion(pred_attack["pred_adv"].to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                    loss_dis = pred_attack["loss_dis"]
                    loss_reg = pred_attack["adv_loss_reg"]
                    node_adv_list.append(pred_attack["node_adv"])
                    edge_adv_list.append(pred_attack["edge_adv"])
                    loss = loss_adv - loss_dis * args.adv_dis - loss_reg * args.adv_reg
                    (-loss).backward()
                    total_loss_adv += loss_adv.item()
                    total_loss_dis += loss_dis.item()
                    total_loss_reg += loss_reg.item()
                optimizer.step()
                total_loss += loss.item()
            if step % args.show == 0:
                if optimizer_name == 'causaler':
                    print("{} | Epoch:[{}/{}] Train Iter:[{:<3}/{}] Loss (cau,com,reg):[{:.4f} ({:.4f},{:.4f},{:.4f})], Cau n/e:[{:.2f}/{:.2f}] Adv n/e:[{:.2f}/{:.2f}]"
                            .format(optimizer_name, epoch, args.epochs, step, len(train_loader), 
                                    total_loss / (step + 1),
                                    total_loss_cau / (step + 1),
                                    total_loss_com / (step + 1),
                                    total_loss_reg / (step + 1),
                                    np.mean(node_cau_list),
                                    np.mean(edge_cau_list),
                                    np.mean(node_adv_list),
                                    np.mean(edge_adv_list)))
                else:
                    print("{} | Epoch:[{}/{}] Train Iter:[{:<3}/{}] Loss (adv,dis,reg):[{:.4f} ({:.4f},{:.4f},{:.4f})], Adv n/e:[{:.2f}/{:.2f}]"
                            .format(optimizer_name, epoch, args.epochs, step, len(train_loader), 
                                    total_loss / (step + 1),
                                    total_loss_adv / (step + 1),
                                    total_loss_dis / (step + 1),
                                    total_loss_reg / (step + 1),
                                    np.mean(node_adv_list),
                                    np.mean(edge_adv_list)))
        epoch_loss = total_loss / len(train_loader)
        if optimizer_name == 'causaler':
            node_cau = np.mean(node_cau_list)
            edge_cau = np.mean(edge_cau_list)
        node_adv = np.mean(node_adv_list)
        edge_adv = np.mean(edge_adv_list)
        #################################### End training #####################################
        valid_result = eval(model, device, valid_loader, evaluator)[dataset.eval_metric]
        test_result = eval(model, device, test_loader, evaluator)[dataset.eval_metric]
        if valid_result > results['highest_valid']:
            results['highest_valid'] = valid_result
            results['update_test'] = test_result
            results['update_epoch'] = epoch
        print("-" * 150)
        if optimizer_name == 'causaler':
            print("{} | Epoch:[{}/{}], loss:[{:.4f}], valid:[{:.2f}], test:[{:.2f}] | Best val:[{:.2f}] Update test:[{:.2f}] at:[{}] | Cau n/e:[{:.2f}/{:.2f}], Adv n/e:[{:.2f}/{:.2f}] | Time:{:.2f} min"
                            .format(optimizer_name,epoch, args.epochs, epoch_loss, 
                                    valid_result * 100,
                                    test_result * 100,
                                    results['highest_valid'] * 100,
                                    results['update_test'] * 100,
                                    results['update_epoch'],
                                    node_cau,
                                    edge_cau,
                                    node_adv,
                                    edge_adv,
                                    (time.time()-start_time_local) / 60))
        else:
            print("{} | Epoch:[{}/{}], loss:[{:.4f}], valid:[{:.2f}], test:[{:.2f}] | Best val:[{:.2f}] Update test:[{:.2f}] at:[{}] | Adv n/e:[{:.2f}/{:.2f}] | Time:{:.2f} min"
                            .format(optimizer_name,epoch, args.epochs, epoch_loss, 
                                    valid_result * 100,
                                    test_result * 100,
                                    results['highest_valid'] * 100,
                                    results['update_test'] * 100,
                                    results['update_epoch'],
                                    node_adv,
                                    edge_adv,
                                    (time.time()-start_time_local) / 60))
        print("-" * 150)
    total_time = time.time() - start_time
    print("syd: Best val:[{:.2f}] Update test:[{:.2f}] at epoch:[{}] | Total time:{}"
            .format(results['highest_valid'] * 100,
                    results['update_test'] * 100,
                    results['update_epoch'],
                    time.strftime('%H:%M:%S', time.gmtime(total_time))))
    return results['highest_valid'], results['update_test']

def config_and_run(args):
    
    print_args(args)
    final_test_acc = []
    for _ in range(args.trails):
        valid_auc, test_auc = main(args)
        final_test_acc.append(test_auc)
    print('sydfinall: Test ACC: {:.2f} ± {:.2f}, ALL: {}'.format(np.mean(final_test_acc) * 100, np.std(final_test_acc) * 100, final_test_acc))

if __name__ == "__main__":

    def arg_parse():
        str2bool = lambda x: x.lower() == "true"
        parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
        parser.add_argument('--fro_layer', type=int,   default=2,   help='dropout ratio (default: 0.5)')
        parser.add_argument('--bac_layer', type=int,   default=2,   help='dropout ratio (default: 0.5)')
        parser.add_argument('--cau_layer', type=int,   default=2,   help='dropout ratio (default: 0.5)')
        parser.add_argument('--att_layer', type=int,   default=2,   help='dropout ratio (default: 0.5)')
        parser.add_argument('--cau_gamma', type=float, default=0.4, help='dropout ratio (default: 0.5)')
        parser.add_argument('--adv_gamma',  type=float,     default=1.0,         help='dropout ratio (default: 0.5)')
        parser.add_argument('--adv_dis',    type=float,     default=0.5,         help='dropout ratio (default: 0.5)')
        parser.add_argument('--adv_reg',    type=float,     default=0.05,        help='dropout ratio (default: 0.5)')
        parser.add_argument('--cau_reg',    type=float,     default=0.05,        help='dropout ratio (default: 0.5)')
        parser.add_argument('--causaler_time', type=int,    default=1,           help='path for alternative optimization')
        parser.add_argument('--attacker_time', type=int,    default=1,           help='path for alternative optimization')
        parser.add_argument('--by_default', action='store_true')

        parser.add_argument('--causaler_lr',     type=float, default=1e-3, help='Learning rate (default: 1e-2)')
        parser.add_argument('--attacker_lr',     type=float, default=1e-3, help='Learning rate (default: 1e-2)')

        parser.add_argument('--l2reg',  type=float, default=5e-6, help='L2 norm (default: 5e-6)')
        parser.add_argument('--device', type=int, default=0, help='which gpu')
        parser.add_argument('--show', type=int, default=10, help='which gpu')
        parser.add_argument('--gnn', type=str, default='gin', help='gin or gcn')
        parser.add_argument('--drop_ratio', type=float, default=0.5, help='dropout ratio (default: 0.5)')
        parser.add_argument('--num_layer',      type=int, default=5, help='number of GNN message passing layers (default: 5)')
        parser.add_argument('--emb_dim',        type=int, default=300, help='dimensionality of hidden units in GNNs (default: 300)')
        parser.add_argument('--batch_size',     type=int, default=512, help='input batch size for training (default: 32)')
        parser.add_argument('--epochs',         type=int, default=100, help='number of epochs to train (default: 100)')
        parser.add_argument('--num_workers', type=int, default=0, help='number of workers (default: 0)')
        parser.add_argument('--dataset', type=str, default="ogbg-molhiv", help='dataset name (default: ogbg-molhiv)')
        parser.add_argument('--trails', type=int, default=1, help='number of runs (default: 0)')     

        args = parser.parse_args()
        return args

    args = arg_parse()
    config_and_run(args)
