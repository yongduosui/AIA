from GOOD import config_summoner
from GOOD.utils.args import args_parser
from GOOD.utils.logger import load_logger
from GOOD.kernel.pipeline import initialize_model_dataset
from GOOD.ood_algorithms.ood_manager import load_ood_alg
from GOOD.kernel.pipeline import load_task
import numpy as np
import pdb

def run(config):

    model, loader = initialize_model_dataset(config)
    ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
    test_iid, test_ood = load_task(config.task, model, loader, ood_algorithm, config)
    return test_iid, test_ood

def main():

    iid_list = []
    ood_list = []
    run_time = 10
    args = args_parser()
    config = config_summoner(args)
    load_logger(config)
    for i in range(run_time):
        test_iid, test_ood = run(config)
        print("syd: run:{}/{}, test iid:{:.2f}, test ood:{:.2f}".format(i + 1, run_time, test_iid, test_ood))
        iid_list.append(test_iid)
        ood_list.append(test_ood)

    print('sydfinall: Test IID:{:.2f} ± {:.2f}, Test OOD:{:.2f} ± {:.2f}'
        .format(np.mean(iid_list), np.std(iid_list), 
                np.mean(ood_list), np.std(ood_list)))
main()