import os
import random
import shlex
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Any, Callable, DefaultDict, Iterable, List, Optional, Sequence, Tuple
from torch_geometric.loader import DataLoader
import numpy as np
import torch
import torch.utils.data as data
import pdb
from typing import Sequence

import sklearn
from sklearn.preprocessing import StandardScaler
import scipy
from scipy.stats import gaussian_kde
from torch_geometric.utils.dropout import dropout_adj
import warnings
warnings.filterwarnings('ignore')

class Task:
    def launch(self, worker_index: int):
        raise NotImplementedError
        
    def on_start(self):
        raise NotImplementedError
        
    def on_completion(self):
        raise NotImplementedError
        
    def on_failure(self):
        raise NotImplementedError
    
    
class ParallelRunner:
    def __init__(self, max_workers: int, worker_type: str) -> None:
        self.max_workers = max_workers
        self.worker_type = worker_type
        
    def launch(self, tasks: Sequence[Task], callback: Callable = None) -> None:
        self.num_tasks = len(tasks)
        self.completed, self.failed = [], []
        self.task_futures = [None] * self.max_workers
        self.next = 0
        if self.worker_type == 'thread':
            Executor = ThreadPoolExecutor
        elif self.worker_type == 'process':
            Executor = ProcessPoolExecutor
        else:
            raise NotImplementedError
        self.executor = Executor(max_workers=self.max_workers)
        while len(self.completed) + len(self.failed) < len(tasks):
            for worker in self.idle_workers:
                self.check_completion(worker)
                if self.next == len(tasks):
                    continue
                self.assign(worker, tasks[self.next], callback)
                self.next += 1
            time.sleep(1)
        self.executor.shutdown()
        
    def check_completion(self, worker) -> None:
        if not self.task_futures[worker]:
            return
        task, future = self.task_futures[worker]
        if future.exception():
            task.on_failure()
            self.failed.append(future)
        else:
            task.on_completion()
            self.completed.append(future)
        self.task_futures[worker] = None
        print(f'{len(self.completed)} completed, {len(self.failed)} failed, '
              f'{self.num_tasks - len(self.completed) - len(self.failed)} to go')
        
    def assign(self, worker: int, task: Task, callback: Callable) -> None:
        future = self.executor.submit(task.start, worker)
        task.on_start()
        if callback:
            future.add_done_callback(callback)
        self.task_futures[worker] = (task, future)
        
    @property
    def idle_workers(self) -> List[int]:
        idle_workers = []
        for i in range(self.max_workers):
            if self.task_futures[i]:
                task, future = self.task_futures[i]
                if future.done():
                    idle_workers.append(i)
            else:
                idle_workers.append(i)
        return idle_workers



def get_label_map(dataset):

    label_maps = defaultdict(list)
    for i, label in enumerate(dataset.data.y.tolist()):
        label_maps[label].append(i)
    return label_maps

def get_data_list_label_map(data_list):

    label_maps = defaultdict(list)
    for i, data in enumerate(data_list):
        label = data.y.item()
        label_maps[label].append(i)
    
    return label_maps

def get_subset_list(dataset, index_list, dropedge=0):

    data_list = [dataset[i] for i in index_list]
    if dropedge > 0:
        orig_edge_list = []
        news_edge_list = []
        for data in data_list:
            orig_edge_list.append(data.num_edges)
            data.edge_index, _ = dropout_adj(data.edge_index, p=dropedge)
            news_edge_list.append(data.num_edges)
        print("[!] drop:{:.2f}%".format(100 - 100 * sum(news_edge_list) / sum(orig_edge_list)))
    return data_list

def split_dataset(dataset, fraction, seed):
    """ Randomly split a dataset with deterministic label distribution. """
    
    label_map = get_data_list_label_map(dataset)
    rng = np.random.RandomState(seed)
    keys_p = []
    keys_q = []
    for _, keys in label_map.items():
        n = int(len(keys) * fraction)
        rng.shuffle(keys)
        keys_p += keys[:n]
        keys_q += keys[n:]
    
    subset1 = get_subset_list(dataset, keys_p)
    subset2 = get_subset_list(dataset, keys_q)
    return subset1, subset2
    # return dataset[keys_p], dataset[keys_q]
    # return Subset_(dataset, keys_p), Subset_(dataset, keys_q)
    
def create_label_shift_free_datasets(datasets, seed):
    label_maps = [get_label_map(dataset) for dataset in datasets]
    
    if not label_maps:
        raise RuntimeError
    labels = set(label_maps[0])

    min_count = {}
    for label in labels:
        min_count[label] = min(len(map_[label]) for map_ in label_maps)

    keys_list = [[] for _ in range(len(label_maps))]
    rng = np.random.RandomState(seed)
    for label, count in min_count.items():
        for i, map_ in enumerate(label_maps):
            rng.shuffle(map_[label])
            keys_list[i] += map_[label][:count]

    return keys_list[0], keys_list[1]
    
def get_dataloaders(dataset, holdout, args, dropedge):

    seed = args.seed
    batch_size = args.batch_size

    if dropedge > 0:
        print("[!] dropedge!:{:.1f}".format(dropedge))
    
    n_workers = 0
    dataset_p, dataset_q = dataset["train"], dataset["test"]
    subset_p_index, subset_q_index = create_label_shift_free_datasets([dataset_p, dataset_q], seed)

    subset_p = get_subset_list(dataset_p, subset_p_index, dropedge)
    subset_q = get_subset_list(dataset_q, subset_q_index)
    
    def in_out_split(dataset):
        
        out, in_ = split_dataset(dataset, holdout, seed)
        in_dataloader  = DataLoader(in_, batch_size, shuffle=True,
                                         num_workers=n_workers, drop_last=True)
        out_dataloader = DataLoader(out, batch_size, shuffle=False,
                                         num_workers=n_workers)
        return in_dataloader, out_dataloader

    in_dataloader_p, out_dataloader_p = in_out_split(subset_p)
    in_dataloader_q, out_dataloader_q = in_out_split(subset_q)
    
    print(f'Envs p: #datapoints (in+out): {len(subset_p)} '
          f'({len(in_dataloader_p.dataset)}+{len(out_dataloader_p.dataset)})')
    print(f'Envs q: #datapoints (in+out): {len(subset_q)} '
          f'({len(in_dataloader_q.dataset)}+{len(out_dataloader_q.dataset)})')
    
    return in_dataloader_p, out_dataloader_p, in_dataloader_q, out_dataloader_q



    
def set_deterministic(seed: int = 0):
    ''' Reference: https://pytorch.org/docs/stable/notes/randomness.html '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        
class gaussian_kde_(gaussian_kde):
    """ A gaussian kde that is friendly to small samples. """
    def _compute_covariance(self) -> None:
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        self.factor = self.covariance_factor()
        # Cache covariance and inverse covariance of the data
        if not hasattr(self, '_data_inv_cov'):
            self._data_covariance = np.atleast_2d(
                np.cov(self.dataset, rowvar=1, bias=False, aweights=self.weights))
            w, v = np.linalg.eigh(self._data_covariance)
            # Set near-zero eigenvalues to a small number, avoiding singular covariance
            # matrices when the sample do not span the whole feature space
            w[np.where(abs(w) < 1e-9)[0]] = 0.01
            self._data_inv_cov = np.linalg.inv(v @ np.diag(w) @ v.T)

        self.covariance = self._data_covariance * self.factor**2
        self.inv_cov = self._data_inv_cov / self.factor**2


def compute_div(p: Sequence[float], q: Sequence[float], probs: Sequence[int],
                eps_div: float) -> float:
    if not len(p) == len(q) == len(probs):
        raise ValueError
    div = 0
    for i in range(len(probs)):
        if p[i] < eps_div or q[i] < eps_div:
            div += abs(p[i] - q[i]) / probs[i]
    div /= len(probs) * 2
    return div


def compute_cor(y_p: np.ndarray, z_p: np.ndarray, y_q: np.ndarray, z_q: np.ndarray,
                p: Sequence[float], q: Sequence[float], probs: Sequence[int],
                points: np.ndarray, eps_cor: float, strict: bool = False) -> float:
    if not len(p) == len(q) == len(probs):
        raise ValueError
    y_p_unique, y_q_unique = map(np.unique, (y_p, y_q))
    if not np.all(y_p_unique == y_q_unique):
        raise ValueError
    classes = sorted(y_p_unique)
    n_classes = len(classes)
    sample_sizes = np.zeros(n_classes, dtype=int)
    cors = np.zeros(n_classes, dtype=float)
    
    for i in range(n_classes):
        y = classes[i]
        indices_p = np.where(y_p == y)[0]
        indices_q = np.where(y_q == y)[0]
        if indices_p.shape != indices_q.shape:
            raise ValueError(f'Number of datapoints mismatch (y={y}): '
                             f'{indices_p.shape} != {indices_q.shape}')
        try:
            kde_p = gaussian_kde_(z_p[indices_p].T)
            kde_q = gaussian_kde_(z_q[indices_q].T)
            p_given_y = kde_p(points)
            q_given_y = kde_q(points)
        except (np.linalg.LinAlgError, ValueError) as exception:
            if strict:
                raise exception
            print(f'WARNING: skipping y={y} because scipy.stats.gaussian_kde '
                  f'failed. This usually happens when there is too few datapoints.')
            print(f'y={y}: #datapoints=({len(indices_p)}, {len(indices_q)}), '
                  f'skipped')
            continue
        sample_sizes[i] = len(indices_p)
        
        for j in range(len(probs)):
            if p[j] > eps_cor and q[j] > eps_cor:
                integrand = abs(p_given_y[j] * np.sqrt(q[j] / p[j])
                              - q_given_y[j] * np.sqrt(p[j] / q[j]))
                cor_j = integrand / probs[j]
            else:
                integrand = cor_j = 0
            cors[i] += cor_j
        cors[i] /= len(probs) * 2
        print(f'y={y}: #datapoints=({len(indices_p)}, {len(indices_q)}), '
              f'value={cors[i]:.4f}')
    cor = np.sum(sample_sizes * cors) / np.sum(sample_sizes)
    return cor

