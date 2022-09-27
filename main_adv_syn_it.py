import torch
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import torch.optim as optim
import torch.nn.functional as F
import argparse
import time
import numpy as np
from GOOD.data.good_datasets.good_cmnist import GOODCMNIST
from GOOD.data.good_datasets.good_motif import GOODMotif
from models import CausalAdvGNNSyn
import matplotlib.pyplot as plt
import matplotlib
import pdb
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
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model.forward_causal(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)

def main(args):
    
    file_name = "{}-adv".format(args.dataset)
    eval_metric = "rocauc"
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    
    if args.dataset == "cmnist":
        dataset, meta_info = GOODCMNIST.load("dataset", domain=args.domain, shift='covariate', generate=False)
        num_class, num_layer, in_dim = 10, 3, 3
    elif args.dataset == "motif":
        dataset, meta_info = GOODMotif.load("dataset", domain=args.domain, shift='covariate', generate=False)
        num_class, num_layer, in_dim = 3, 3, 1
    else:
        assert False
    train_loader     = DataLoader(dataset["train"],  batch_size=args.batch_size, shuffle=True)
    valid_loader_ood = DataLoader(dataset["val"],    batch_size=args.batch_size, shuffle=False)
    valid_loader_iid = DataLoader(dataset["id_val"], batch_size=args.batch_size, shuffle=False)
    test_loader_ood  = DataLoader(dataset["test"],   batch_size=args.batch_size, shuffle=False)
    test_loader_iid  = DataLoader(dataset["id_test"],batch_size=args.batch_size, shuffle=False)

    model = CausalAdvGNNSyn(num_class=num_class, 
                            in_dim=in_dim,
                            emb_dim=300,
                            fro_layer=args.fro_layer,
                            bac_layer=args.bac_layer,
                            cau_layer=args.cau_layer,
                            att_layer=args.att_layer, 
                            cau_gamma=args.cau_gamma,
                            adv_gamma=args.adv_gamma).to(device)
    

    opt_attacker = optim.Adam(model.attacker.parameters(), lr=args.attacker_lr, weight_decay=args.l2reg)
    opt_causaler = optim.Adam(list(model.graph_front.parameters()) 
                            + list(model.graph_backs.parameters()) 
                            + list(model.causaler.parameters())
                            + list(model.predictor.parameters()), 
                              lr=args.causaler_lr,
                              weight_decay=args.l2reg)

    criterion = torch.nn.CrossEntropyLoss()
    results = {'highest_valid_ood': 0, 
               'highest_valid_iid': 0,
               'update_test_ood': 0,  
               'update_test_iid': 0, 
               'update_epoch_ood': 0, 
               'update_epoch_iid': 0}
    start_time = time.time()

    causaler_branch = [model.graph_front, model.graph_backs, model.causaler, model.predictor]
    attacker_branch = [model.attacker]
    test_list_ood, test_list_iid = [], []
    for epoch in range(args.epochs):

        start_time_local = time.time()
        total_loss_causaler, total_loss_attacker = 0, 0
        total_loss_cau, total_loss_com, total_loss_reg, total_loss_adv, total_loss_dis = 0,0,0,0,0
        node_cau_list, edge_cau_list, node_adv_list, edge_adv_list  = [], [], [], []
        
        show = int(float(len(train_loader)) / 4.0)
        
        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            set_network_train_eval(causaler_branch, attacker_branch, train="causaler")
            pred_advcausal = model.forward_advcausal(batch)
            loss_cau = criterion(pred_advcausal["pred_cau"], batch.y.long())
            loss_com = criterion(pred_advcausal["pred_com"], batch.y.long())
            loss_reg = pred_advcausal["cau_loss_reg"]
            
            node_cau_list.append(pred_advcausal["node_cau"])
            edge_cau_list.append(pred_advcausal["edge_cau"])
            loss_causaler = loss_cau + loss_com + loss_reg * args.cau_reg
            loss_causaler.backward()
            
            opt_causaler.step()
            total_loss_causaler += loss_causaler.item()
            total_loss_cau += loss_cau.item()
            total_loss_com += loss_com.item() 
            total_loss_reg += loss_reg.item()

            set_network_train_eval(causaler_branch, attacker_branch, train="attacker")
            pred_attack = model.forward_attack(batch)
            loss_adv = criterion(pred_attack["pred_adv"], batch.y.long())
            loss_dis = pred_attack["loss_dis"]
            loss_reg = pred_attack["adv_loss_reg"]

            node_adv_list.append(pred_attack["node_adv"])
            edge_adv_list.append(pred_attack["edge_adv"])
            
            loss_attacker = loss_adv - loss_dis * args.adv_dis - loss_reg * args.adv_reg
            (-loss_attacker).backward()
            
            opt_attacker.step()
            total_loss_attacker += loss_attacker.item()
            total_loss_adv += loss_adv.item()
            total_loss_dis += loss_dis.item()
            total_loss_reg += loss_reg.item()
        
            if step % show == 0:
                print("Epoch:[{}/{}] Train Iter:[{:<3}/{}] CauLoss (cau,com,reg):[{:.4f} ({:.4f},{:.4f},{:.4f})], AdvLoss (adv,dis,reg):[{:.4f} ({:.4f},{:.4f},{:.4f})] | Cau n/e:[{:.2f}/{:.2f}] Adv n/e:[{:.2f}/{:.2f}]"
                        .format(epoch, args.epochs, 
                                step, len(train_loader), 
                                total_loss_causaler / (step + 1),
                                total_loss_cau / (step + 1),
                                total_loss_com / (step + 1),
                                total_loss_reg / (step + 1),
                                total_loss_attacker / (step + 1),
                                total_loss_adv / (step + 1),
                                total_loss_dis / (step + 1),
                                total_loss_reg / (step + 1),
                                np.mean(node_cau_list),
                                np.mean(edge_cau_list),
                                np.mean(node_adv_list),
                                np.mean(edge_adv_list)))
                break
                
        epoch_loss_causaler = total_loss_causaler / len(train_loader)
        epoch_loss_attacker = total_loss_attacker / len(train_loader)
        node_cau, edge_cau = np.mean(node_cau_list), np.mean(edge_cau_list)
        node_adv, edge_adv = np.mean(node_adv_list), np.mean(edge_adv_list)
        #################################### End training #####################################
        valid_result_iid = eval(model, valid_loader_iid, device)
        valid_result_ood = eval(model, valid_loader_ood, device)
        test_result_iid  = eval(model, test_loader_iid,  device)
        test_result_ood  = eval(model, test_loader_ood,  device)

        if valid_result_iid > results['highest_valid_iid'] and epoch > args.test_epoch:
            results['highest_valid_iid'] = valid_result_iid
            results['update_test_iid'] = test_result_iid
            results['update_epoch_iid'] = epoch
        if valid_result_ood > results['highest_valid_ood'] and epoch > args.test_epoch:
            results['highest_valid_ood'] = valid_result_ood
            results['update_test_ood'] = test_result_ood
            results['update_epoch_ood'] = epoch

        test_list_ood.append(test_result_ood)
        test_list_iid.append(test_result_iid)
        # plot_test(test_list_iid, test_list_ood, file_name)
        plot_test(test_list_ood, file_name)
        print("-" * 150)
        epoch_time = (time.time()-start_time_local) / 60
        print("Epoch:[{}/{}], CauLoss:[{:.4f}] AdvLoss:[{:.4f}], valid:[{:.2f}/{:.2f}], test:[{:.2f}/{:.2f}] | Best val:[{:.2f}/{:.2f}] Update test:[{:.2f}/{:.2f}] at:[{}/{}] | Cau n/e:[{:.2f}/{:.2f}], Adv n/e:[{:.2f}/{:.2f}] | Time:{:.2f} min"
                        .format(epoch, args.epochs, 
                                epoch_loss_causaler, 
                                epoch_loss_attacker,
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
                                node_cau, edge_cau, node_adv, edge_adv, epoch_time))
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
    set_seed(args.seed)
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
        parser.add_argument('--seed',      type=int,   default=666,   help='dropout ratio (default: 0.5)')
        parser.add_argument('--test_epoch', type=int, default=20)
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
        parser.add_argument('--epochs', type=int, default=100)

        parser.add_argument('--causaler_lr',     type=float, default=1e-3, help='Learning rate (default: 1e-2)')
        parser.add_argument('--attacker_lr',     type=float, default=1e-3, help='Learning rate (default: 1e-2)')
        parser.add_argument('--l2reg',           type=float, default=5e-6, help='L2 norm (default: 5e-6)')

        parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
        parser.add_argument('--domain', type=str, default='color', help='color, background')
        parser.add_argument('--batch_size', type=int, default=32)
        
        parser.add_argument('--dataset', type=str, default="hiv")
        parser.add_argument('--trails', type=int, default=1, help='number of runs (default: 0)')   
        args = parser.parse_args()
        return args

    args = arg_parse()
    config_and_run(args)
