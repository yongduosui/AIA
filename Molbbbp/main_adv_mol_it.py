import torch
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import torch.optim as optim
import torch.nn.functional as F
import argparse
import time
import numpy as np
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
import os
from models import CausalAdvGNNMol
import pdb
import random
from utils import *

def init_weights(net, init_type='orthogonal', init_gain=0.02):
    
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'default':
                pass
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
    
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


def set_network_train_eval(causaler, attacker, train="causaler"):
    
    if train == 'causaler':
        for net in causaler:
            net.train()
            net.zero_grad()
        for net in attacker:
            net.eval()
    else:
        for net in attacker:
            net.train()
            net.zero_grad()
        for net in causaler:
            net.eval()
        
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
                pred = model.forward_causal(batch)
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    output = evaluator.eval(input_dict)
    return output

def main(args):
    
    device = torch.device("cuda:0")
    dataset = PygGraphPropPredDataset(name=args.dataset, root=args.data_dir)
    num_class = dataset.num_tasks
    eval_metric = "rocauc"

    if args.domain == "scaffold":
        split_idx = dataset.get_idx_split()
    else:
        split_idx = size_split_idx(dataset, args.size)

    evaluator = Evaluator(args.dataset)
    train_loader     = DataLoader(dataset[split_idx["train"]],  batch_size=args.batch_size, shuffle=True,  num_workers=0, worker_init_fn=_init_fn)
    valid_loader_ood = DataLoader(dataset[split_idx["valid"]],  batch_size=args.batch_size, shuffle=False, num_workers=0, worker_init_fn=_init_fn)
    test_loader_ood  = DataLoader(dataset[split_idx["test"]],   batch_size=args.batch_size, shuffle=False, num_workers=0, worker_init_fn=_init_fn)
    model = CausalAdvGNNMol(num_class=num_class, 
                            emb_dim=args.emb_dim, 
                            cau_gamma=args.cau_gamma,
                            adv_gamma_node=args.adv_gamma_node,
                            adv_gamma_edge=args.adv_gamma_edge).to(device)
    init_weights(model, args.initw_name, init_gain=0.02)
    opt_attacker = optim.Adam(model.attacker.parameters(), lr=args.attacker_lr, weight_decay=args.l2reg)
    opt_causaler = optim.Adam(list(model.graph_front.parameters()) 
                            + list(model.graph_backs.parameters()) 
                            + list(model.causaler.parameters())
                            + list(model.predictor.parameters()), 
                              lr=args.causaler_lr, weight_decay=args.l2reg)
    if args.lr_scheduler == 'step':
        sch_attacker = StepLR(opt_attacker, step_size=args.lr_decay, gamma=args.lr_gamma)
        sch_causaler = StepLR(opt_causaler, step_size=args.lr_decay, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'muti':
        sch_attacker = MultiStepLR(opt_attacker, milestones=args.milestones, gamma=args.lr_gamma)
        sch_causaler = MultiStepLR(opt_causaler, milestones=args.milestones, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cos':
        sch_attacker = CosineAnnealingLR(opt_attacker, T_max=args.epochs, eta_min=args.lr_min)
        sch_causaler = CosineAnnealingLR(opt_causaler, T_max=args.epochs, eta_min=args.lr_min)
    else:
        pass

    criterion = torch.nn.BCEWithLogitsLoss()
    results = {'highest_valid': 0, 'update_test': 0,  'update_epoch': 0}
    start_time = time.time()

    causaler_branch = [model.graph_front, model.graph_backs, model.causaler, model.predictor]
    attacker_branch = [model.attacker]

    for epoch in range(1, args.epochs + 1):

        start_time_local = time.time()
        total_loss_causaler, total_loss_attacker = 0, 0

        show = int(float(len(train_loader)) / 4.0)
        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            set_network_train_eval(causaler_branch, attacker_branch, train="causaler")
            pred_advcausal = model.forward_advcausal(batch)
            is_labeled = batch.y == batch.y
            ground_truth = batch.y.to(torch.float32)[is_labeled]
            loss_cau = criterion(pred_advcausal["pred_cau"].to(torch.float32)[is_labeled], ground_truth)
            loss_com = criterion(pred_advcausal["pred_com"].to(torch.float32)[is_labeled], ground_truth)
            loss_reg = pred_advcausal["cau_loss_reg"]
            
            
            loss_causaler = loss_cau + loss_com + loss_reg * args.cau_reg
            loss_causaler.backward()
            
            opt_causaler.step()
            total_loss_causaler += loss_causaler.item()

            set_network_train_eval(causaler_branch, attacker_branch, train="attacker")
            pred_attack = model.forward_attack(batch)
            loss_adv = criterion(pred_attack["pred_adv"].to(torch.float32)[is_labeled], ground_truth)
            loss_dis = pred_attack["loss_dis"]
            loss_reg = pred_attack["adv_loss_reg"]
            
            loss_attacker = loss_adv - loss_dis * args.adv_dis - loss_reg * args.adv_reg
            (-loss_attacker).backward()
            
            opt_attacker.step()
            total_loss_attacker += loss_attacker.item()

            if show != 0 and step % show == 0:
                print("Train Iter:[{:<3}/{}] CauLoss:[{:.4f}], AdvLoss:[{:.4f}]"
                        .format(step, len(train_loader), 
                                total_loss_causaler / (step + 1),
                                total_loss_attacker / (step + 1)))

        epoch_loss_causaler = total_loss_causaler / len(train_loader)
        epoch_loss_attacker = total_loss_attacker / len(train_loader)

        #################################### End training #####################################
        valid_result = eval(model, evaluator, valid_loader_ood, device)[eval_metric]
        test_result  = eval(model, evaluator, test_loader_ood,  device)[eval_metric]

        if valid_result > results['highest_valid'] and epoch > args.test_epoch:
            results['highest_valid'] = valid_result
            results['update_test'] = test_result
            results['update_epoch'] = epoch

        if args.lr_scheduler in ['step', 'muti', 'cos']:
            sch_causaler.step()
            sch_attacker.step()
        print("-" * 200)
        epoch_time = (time.time()-start_time_local) / 60
        print("Epoch:[{}/{}], CauAdvLoss:[{:.4f}/{:.4f}] valid:[{:.2f}], test:[{:.2f}]"
                        .format(epoch, args.epochs, 
                                epoch_loss_causaler, 
                                epoch_loss_attacker,
                                valid_result * 100, 
                                test_result * 100))
        print("-" * 200)

    total_time = time.time() - start_time
    print("Update test:[{:.2f}] at epoch:[{}] | Total time:{}"
            .format(results['update_test'] * 100,
                    results['update_epoch'],
                    time.strftime('%H:%M:%S', time.gmtime(total_time))))
    return results['update_test']

def config_and_run(args):
    
    print_args(args)
    final_test_acc = []
    for _ in range(args.trails):
        args.seed += 10
        set_seed(args.seed)
        # print("seed:{}".format(args.seed))
        test_auc = main(args)
        final_test_acc.append(test_auc)
    print("finall: Test ACC OOD: [{:.2f}±{:.2f}]".format(np.mean(final_test_acc) * 100, np.std(final_test_acc) * 100))
    print("ALL OOD:{}".format(final_test_acc))
if __name__ == "__main__":

    def arg_parse():
        str2bool = lambda x: x.lower() == "true"
        parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')

        parser.add_argument('--initw_name', type=str, default='kaiming',
                        choices=['default','orthogonal','normal','xavier','kaiming'],
                        help='method name to initialize neural weights')

        parser.add_argument('--data_dir',  type=str, default='../../dataset', help='dataset dir')
        parser.add_argument('--ckpt',      type=str2bool,   default=False)
        parser.add_argument('--seed',      type=int,   default=123)
        parser.add_argument('--emb_dim',   type=int,   default=300)
        parser.add_argument('--test_epoch', type=int,  default=0)
        parser.add_argument('--cau_gamma', type=float, default=0.4)
        parser.add_argument('--lr_min',    type=float, default=1e-6)
        # parser.add_argument('--adv_gamma',  type=float,     default=1.0,         help='dropout ratio (default: 0.5)')
        parser.add_argument('--adv_gamma_node',    type=float,     default=1.0)
        parser.add_argument('--adv_gamma_edge',    type=float,     default=1.0)

        parser.add_argument('--adv_dis',    type=float,     default=0.5)
        parser.add_argument('--adv_reg',    type=float,     default=0.05)
        parser.add_argument('--cau_reg',    type=float,     default=0.05)
        parser.add_argument('--causaler_time', type=int,    default=1)
        parser.add_argument('--attacker_time', type=int,    default=1)
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--lr_decay', type=int, default=150)
        parser.add_argument('--causaler_lr',     type=float, default=1e-3, help='Learning rate')
        parser.add_argument('--attacker_lr',     type=float, default=1e-3, help='Learning rate')
        parser.add_argument('--l2reg',           type=float, default=5e-6, help='L2 norm (default: 5e-6)')
        parser.add_argument('--the', type=float,   default=0)

        parser.add_argument('--lr_gamma', type=float, default=0.1)
        parser.add_argument('--lr_scheduler',  type=str, default='cos', help='color, background')
        parser.add_argument('--milestones', nargs='+', type=int, default=[3703,16,6])

        parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
        parser.add_argument('--domain', type=str, default='color', help='color, background')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--size', type=str, default='ls', help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
        parser.add_argument('--dataset', type=str, default="hiv")
        parser.add_argument('--trails', type=int, default=1, help='number of runs (default: 0)')   
        args = parser.parse_args()
        return args

    args = arg_parse()
    config_and_run(args)
