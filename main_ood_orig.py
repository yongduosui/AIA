from GOOD import config_summoner
from GOOD.utils.args import args_parser
from GOOD.utils.logger import load_logger
from GOOD.kernel.pipeline import initialize_model_dataset
from GOOD.ood_algorithms.ood_manager import load_ood_alg
from GOOD.kernel.pipeline import load_task
import pdb

def main():

    args = args_parser()
    config = config_summoner(args)
    load_logger(config)
    model, loader = initialize_model_dataset(config)
    ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
    load_task(config.task, model, loader, ood_algorithm, config)

main()