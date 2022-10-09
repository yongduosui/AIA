import torch
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import torch.optim as optim
import torch.nn.functional as F
import argparse
import time
import numpy as np
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
from GOOD.data.good_datasets.good_cmnist import GOODCMNIST
from GOOD.data.good_datasets.good_motif import GOODMotif
from models import CausalAdvGNNSyn
import matplotlib.pyplot as plt
import matplotlib
import pdb
import os
import random
matplotlib.rcParams.update({'font.size': 18})

def plot_test(test_list_ood, file_name):

    x = [i for i in range(len(test_list_ood))]
    plt.figure(figsize=(8,6))
    plt.grid(linestyle='--')
    # plt.plot(x, test_list_iid, label="test iid")
    plt.plot(x, test_list_ood, label="test ood")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(fontsize=10)
    plt.savefig('{}.png'.format(file_name), bbox_inches='tight')  
    plt.close()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.CEX = False

def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        print(arg + '.' * (str_num - len(arg) - len(str(val))) + str(val))
    print()

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        
        if net is not None:
            for name, param in net.named_parameters():
                param.requires_grad = requires_grad
    
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
        
def eval(model, loader, device):
    
    model.eval()
    correct_cau = 0
    correct_com = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred_cau = model.forward_causal(data).max(1)[1]
            pred_com = model.forward_combined_inference(data).max(1)[1]
        correct_cau += pred_cau.eq(data.y.view(-1)).sum().item()
        correct_com += pred_com.eq(data.y.view(-1)).sum().item()
    correct_cau = correct_cau / len(loader.dataset)
    correct_com = correct_com / len(loader.dataset)
    return correct_cau * 100, correct_com * 100

def main(args):
    
    file_name = "{}-adv".format(args.dataset)
    eval_metric = "rocauc"
    # device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cuda:" + str(args.device))
    # ckpt_path = "ckpts/{}".format(args.dataset + args.ckpt)
    # os.makedirs(ckpt_path, exist_ok=True)
    dataset, meta_info = GOODMotif.load(args.data_dir, domain=args.domain, shift='covariate', generate=False)
    num_class, num_layer, in_dim = 3, 3, 1
    
    train_loader     = DataLoader(dataset["train"],  batch_size=args.batch_size, shuffle=True)
    valid_loader_ood = DataLoader(dataset["val"],    batch_size=args.batch_size, shuffle=False)
    test_loader_ood  = DataLoader(dataset["test"],   batch_size=args.batch_size, shuffle=False)

    model = CausalAdvGNNSyn(num_class=num_class, 
                            in_dim=in_dim,
                            emb_dim=300, 
                            cau_gamma=args.cau_gamma,
                            adv_gamma=args.adv_gamma,
                            adv_gamma_edge=args.adv_gamma_edge).to(device)
    
    opt_attacker = optim.Adam(model.attacker.parameters(), lr=args.attacker_lr, weight_decay=args.l2reg)
    opt_causaler = optim.Adam(list(model.graph_front.parameters()) + list(model.graph_backs.parameters()) + list(model.causaler.parameters()) + list(model.predictor.parameters()), 
                              lr=args.causaler_lr, weight_decay=args.l2reg)

    if args.lr_scheduler == 'step':
        sch_attacker = StepLR(opt_attacker, step_size=args.lr_decay, gamma=args.lr_gamma)
        sch_causaler = StepLR(opt_causaler, step_size=args.lr_decay, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'muti':
        sch_attacker = MultiStepLR(opt_attacker, milestones=args.milestones, gamma=args.lr_gamma)
        sch_causaler = MultiStepLR(opt_causaler, milestones=args.milestones, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cos':
        sch_attacker = CosineAnnealingLR(opt_attacker, T_max=args.epochs)
        sch_causaler = CosineAnnealingLR(opt_causaler, T_max=args.epochs)
    else:
        pass

    criterion = torch.nn.CrossEntropyLoss()
    results = {'highest_valid_cau': 0, 'update_test_cau': 0, 'update_epoch_cau': 0, 'best_acc': 0}
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
            loss_cau = criterion(pred_advcausal["pred_cau"], batch.y.long())
            loss_com = criterion(pred_advcausal["pred_com"], batch.y.long())
            loss_reg = pred_advcausal["cau_loss_reg"]
            
            loss_causaler = loss_cau + loss_com + loss_reg * args.cau_reg
            loss_causaler.backward()
            
            opt_causaler.step()
            total_loss_causaler += loss_causaler.item()
        
            set_network_train_eval(causaler_branch, attacker_branch, train="attacker")
            pred_attack = model.forward_attack(batch)
            loss_adv = criterion(pred_attack["pred_adv"], batch.y.long())
            loss_dis = pred_attack["loss_dis"]
            loss_reg = pred_attack["adv_loss_reg"]

            
            loss_attacker = loss_adv - loss_dis * args.adv_dis - loss_reg * args.adv_reg
            (-loss_attacker).backward()
            
            opt_attacker.step()
            total_loss_attacker += loss_attacker.item()
        
            if step % show == 0:
                print("Train Iter:[{:<3}/{}], CauLoss:[{:.4f}], AdvLoss:[{:.4f}]"
                        .format(step, len(train_loader), 
                                total_loss_causaler / (step + 1),
                                total_loss_attacker / (step + 1)))

        epoch_loss_causaler = total_loss_causaler / len(train_loader)
        epoch_loss_attacker = total_loss_attacker / len(train_loader)
        #################################### End training #####################################
        valid_cau, valid_com = eval(model, valid_loader_ood, device)
        test_cau,  test_com  = eval(model, test_loader_ood,  device)

        if valid_cau > results['highest_valid_cau'] and epoch > args.test_epoch:
            results['highest_valid_cau'] = valid_cau
            results['update_test_cau'] = test_cau
            results['update_epoch_cau'] = epoch
            if args.ckpt:
                torch.save(model.state_dict(), ckpt_path + "/val-best.pt")
        
        if args.lr_scheduler in ["cos", "step", "muti"]:
            sch_causaler.step()
            sch_attacker.step()

        print("-" * 200)
        epoch_time = (time.time()-start_time_local) / 60
        print("Epoch:[{}/{}], CauAdvLoss:[{:.4f}/{:.4f}], valid:[{:.2f}], test:[{:.2f}]"
                        .format(epoch, args.epochs, 
                                epoch_loss_causaler, epoch_loss_attacker,
                                valid_cau, test_cau))
        print("-" * 200)

    total_time = time.time() - start_time
    print("Update test:[{:.2f}] at epoch:[{}] | Total time:{}"
            .format(results['update_test_cau'],
                    results['update_epoch_cau'],
                    time.strftime('%H:%M:%S', time.gmtime(total_time))))
    return results['update_test_cau']

def config_and_run(args):
    
    print_args(args)
    set_seed(args.seed)
    final_test_acc_cau = []
    for _ in range(args.trails):
        args.seed += 10
        set_seed(args.seed)
        test_auc_cau = main(args)
        final_test_acc_cau.append(test_auc_cau)
    print("finall: Test ACC CAU: [{:.2f}±{:.2f}]"
        .format(np.mean(final_test_acc_cau), np.std(final_test_acc_cau)))

if __name__ == "__main__":

    def arg_parse():
        str2bool = lambda x: x.lower() == "true"
        parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')

        parser.add_argument('--data_dir',  type=str, default='../../dataset', help='dataset dir')
        parser.add_argument('--ckpt',      type=str,   default="")
        parser.add_argument('--seed',      type=int,   default=666)
        parser.add_argument('--test_epoch', type=int, default=10)

        parser.add_argument('--cau_gamma', type=float, default=0.4)
        parser.add_argument('--adv_gamma',  type=float,     default=1.0)
        parser.add_argument('--adv_gamma_edge',    type=float,     default=0.8)

        parser.add_argument('--adv_dis',    type=float,     default=0.5)
        parser.add_argument('--adv_reg',    type=float,     default=0.05)
        
        parser.add_argument('--cau_reg',    type=float,     default=0.05)
        parser.add_argument('--causaler_time', type=int,    default=1)
        parser.add_argument('--attacker_time', type=int,    default=1)
        parser.add_argument('--epochs', type=int, default=100)

        parser.add_argument('--the', type=float,   default=0)
        parser.add_argument('--lr_decay', type=int, default=30)
        parser.add_argument('--lr_gamma', type=float, default=0.1)
        parser.add_argument('--lr_scheduler',  type=str, default='step')
        parser.add_argument('--milestones', nargs='+', type=int, default=[40,60,80])

        parser.add_argument('--causaler_lr',     type=float, default=1e-3, help='Learning rate')
        parser.add_argument('--attacker_lr',     type=float, default=1e-3, help='Learning rate')
        parser.add_argument('--l2reg',           type=float, default=5e-6, help='L2 norm (default: 5e-6)')

        parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
        parser.add_argument('--domain', type=str, default='color', help='color, background')
        parser.add_argument('--batch_size', type=int, default=32)
        
        parser.add_argument('--dataset', type=str, default="motif")
        parser.add_argument('--trails', type=int, default=1, help='number of runs (default: 1)')   
        args = parser.parse_args()
        return args

    args = arg_parse()
    config_and_run(args)
