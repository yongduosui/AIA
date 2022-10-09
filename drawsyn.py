import re
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
import pdb
import numpy as np
matplotlib.rcParams.update({'font.size': 18})
import os

def main2():
    
    attacker = nn.Linear(10, 10)
    causaler = nn.Linear(10, 10)
    opt_attacker = optim.Adam(attacker.parameters(), lr=0.005)
    opt_causaler = optim.Adam(causaler.parameters(), lr=0.001)
    sch_attacker = CosineAnnealingLR(opt_attacker, T_max=100)
    sch_causaler = CosineAnnealingLR(opt_causaler, T_max=100)
    lrs_attacker = []
    lrs_causaler = []
    for epoch in range(100):
        lr_cau = opt_causaler.state_dict()['param_groups'][0]['lr']
        lr_adv = opt_attacker.state_dict()['param_groups'][0]['lr']
        lrs_causaler.append(lr_cau)
        lrs_attacker.append(lr_adv)
        sch_attacker.step()
        sch_causaler.step()
    x = [i for i in range(100)]
    plt.figure(figsize=(8,6))
    plt.grid(linestyle='--')
    plt.plot(x, lrs_causaler, label="lr causaler")
    plt.plot(x, lrs_attacker, label="lr attacker")
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.legend(fontsize=18)
    plt.savefig("learning-rates.png", bbox_inches='tight')   
    plt.close()

def get_index(val_list, test_epoch, the):

    best_val = 0
    update_epoch = 0
    for epoch, val in enumerate(val_list):
        if val - best_val > the and epoch > test_epoch:
            best_val = val
            update_epoch = epoch
    return update_epoch

def get_test(file):

    key1 = "Update"
    test_list = []
    val_list = []
    with open(file, "r") as f:
        for line in f.readlines():
            if key1 in line:
                line = line.strip()   
                acc = re.split(r'[:,/,\[,\],|]',line)

                if len(acc) < 20:
                    continue
                # pdb.set_trace()
                test_list.append(float(acc[16]))
                val_list.append(float(acc[12]))
    return test_list, val_list

def draw_file(file, idx_list, mean=False, epochs=100, test_epoch=10, the=0.5):

    update_test_list = []
    test_list, val_list = get_test(file)
    num_drop = int((len(test_list) / epochs)) * epochs
    test_list = torch.tensor(test_list[:num_drop])
    val_list  = torch.tensor(val_list[:num_drop])
    
    if mean:
        test_mean = test_list.view(-1, epochs)[idx_list,:].mean(dim=0).tolist()
        val_mean  = val_list.view(-1, epochs)[idx_list,:].mean(dim=0).tolist()
        test_std = test_list.view(-1, epochs)[idx_list,:].std(dim=0).tolist()
        val_std  = val_list.view(-1, epochs)[idx_list,:].std(dim=0).tolist()
    else:
        test_list = test_list.view(-1, epochs).tolist()
        val_list  = val_list.view(-1, epochs).tolist()

    x = [i for i in range(epochs)]
    plt.figure(figsize=(8,6))
    plt.grid(linestyle='--')
    if mean:
        plt.plot(x, test_mean, label="test-mean", color='blue')
        plt.plot(x, val_mean,  label="val-mean", color='green')
        test_lower = [x - y for x, y in zip(test_mean, test_std)]
        test_upper = [x + y for x, y in zip(test_mean, test_std)]
        val_lower = [x - y for x, y in zip(val_mean, val_std)]
        val_upper = [x + y for x, y in zip(val_mean, val_std)]
        plt.fill_between(x, test_lower, test_upper, color='blue',   alpha=0.2)
        plt.fill_between(x, val_lower,  val_upper,  color='green', alpha=0.2)
        update_idx = get_index(val_mean, test_epoch, the)
        # plt.scatter(x[update_idx], test_mean[update_idx], color='red', marker="o", s=100, zorder=20)
        save_name = '{}-mean.png'.format(file)
        plt.legend(loc='lower right')
    else:
        save_name = '{}-idx.png'.format(file)
        for i, (test_i, val_i) in enumerate(zip(test_list, val_list)):
            if i in idx_list:
                plt.plot(x, test_i, label="test-{}".format(i))
                plt.plot(x, val_i,  label="val-{}".format(i))
                update_idx = get_index(val_i, test_epoch, the)
                plt.scatter(x[update_idx], test_i[update_idx], color='red', marker="o", s=100, zorder=20)
                update_test_list.append(test_i[update_idx])
                print("id:{} update test:{:.2f}".format(i, test_i[update_idx]))
            else:
                continue
        print("Test ACC: [{:.2f}±{:.2f}]".format(np.mean(update_test_list), np.std(update_test_list)))
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig(save_name, bbox_inches='tight')   
    plt.close()


def plot_left(file, epochs=100, test_epoch=10, the=0.5):

    save_name = '{}-left.png'.format(file)
    test_list, val_list = get_test(file)
    total = len(test_list)
    num_drop = int((len(test_list) / epochs)) * epochs
    left = total - num_drop

    test_list = torch.tensor(test_list[num_drop:]).tolist()
    val_list  = torch.tensor(val_list[num_drop:]).tolist()
    
    x = [i for i in range(left)]
    plt.figure(figsize=(8,6))
    plt.grid(linestyle='--')
    plt.plot(x, test_list, label="test-left")
    plt.plot(x, val_list,  label="val-left")
    update_idx = get_index(val_list, test_epoch, the)
    plt.scatter(x[update_idx], test_list[update_idx], color='red', marker="o", s=100, zorder=20)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(fontsize=10)
    plt.savefig(save_name, bbox_inches='tight')   
    plt.close()


def compute_best(file):
    for the in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
        for test_epoch in range(16):
            get_final_acc(file, test_epoch, the)
            
def get_final_acc(file, test_epoch, the):

    epochs = 100
    update_test_list = []
    test_list, val_list = get_test(file)
    test_list = torch.tensor(test_list)
    val_list  = torch.tensor(val_list)
    test_list = test_list.view(-1, epochs).tolist()
    val_list  = val_list.view(-1, epochs).tolist()
    x = [i for i in range(epochs)]
    for i, (test_i, val_i) in enumerate(zip(test_list, val_list)):
        update_idx = get_index(val_i, test_epoch, the)
        update_test_list.append(test_i[update_idx])
    print("the:{} test_epoch:{} Test ACC: [{:.2f}±{:.2f}]".format(the, test_epoch, np.mean(update_test_list), np.std(update_test_list)))

def main():

    root = ""
    file_list = ["10091132-motif-basis-0.8-reproduce.log"]
    # file_list = ["10082158-motif-size-reproduce.log"]

    epochs = 100
    test_epoch = 20
    the = 0

    print("-" * 100)
    print("plot left")
    print("-" * 100)
    for i in [0]:
        draw_name = root + file_list[i]
        idx_list = [i for i in range(9)]
        print("-" * 100)
        print("plot:{}".format(draw_name))
        os.system('cat {} | grep syd'.format(draw_name))
        print("-" * 100)
        draw_file(draw_name, idx_list, mean=False, epochs=epochs, test_epoch=test_epoch, the=the)
        # draw_file(draw_name, idx_list, mean=True, epochs=epochs, test_epoch=test_epoch, the=the)

    # print("-" * 100)
    # print("plot left")
    # for i in [0, 1, 2, 3]:
    #     draw_name = root + file_list[i]
    #     print("plot:{}".format(draw_name))
    #     plot_left(draw_name, test_epoch=test_epoch, epochs=epochs, the=the)

if __name__ == '__main__':
    main()