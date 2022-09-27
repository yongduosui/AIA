import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pdb
import numpy as np
matplotlib.rcParams.update({'font.size': 18})


def get_test(file):

    key1 = "Update"
    test_list = []
    with open(file, "r") as f:
        for line in f.readlines():
            if key1 in line:
                line = line.strip()   
                acc = re.split(r'[:,/,\[,\],|]',line)
                if len(acc) < 20:
                    continue
                test_list.append(float(acc[20]))
    return test_list




def main():

    file_list = ["07101418_setting-it-charmed.log"]
    test_list0 = get_test(file_list[0])[:600]
    test_list0 = torch.tensor(test_list0).view(6, 100).mean(dim=0).tolist()
    test_list1 = get_test(file_list[0])[:100]
    test_list2 = get_test(file_list[0])[100:200]
    test_list3 = get_test(file_list[0])[200:300]
    test_list4 = get_test(file_list[0])[300:400]
    test_list5 = get_test(file_list[0])[400:500]
    test_list6 = get_test(file_list[0])[500:600]



    x = [i for i in range(100)]
    plt.figure(figsize=(8,6))
    plt.grid(linestyle='--')
    plt.plot(x, test_list0, label="test0")
    # plt.plot(x, test_list1, label="test1")
    # plt.plot(x, test_list2, label="test2")
    # plt.plot(x, test_list3, label="test3")
    # plt.plot(x, test_list4, label="test4")
    # plt.plot(x, test_list5, label="test5")
    # plt.plot(x, test_list6, label="test6")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(fontsize=10)
    plt.savefig('motif-it-charmed-mean.png', bbox_inches='tight')   
    plt.close()



if __name__ == '__main__':
    main()
    