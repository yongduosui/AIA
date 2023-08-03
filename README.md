# AIA
We provide a detailed code for AIA.

## Installations
Main packages: PyTorch, Pytorch Geometric, OGB.
```
pytorch==1.10.1
torch-cluster==1.5.9
torch-geometric==2.0.3
torch-scatter==2.0.9
torch-sparse==0.6.12
torch-spline-conv==1.2.1
ogb==1.3.4
```

## Preparations
Please download the graph OOD datasets and OGB datasets as described in the original paper. 
Create a folder ```dataset```, and then put the datasets into ```dataset```. Then modify the path by specifying ```--data_dir your/path/dataset```.


## Commands
 We use the NVIDIA GeForce RTX 3090 (24GB GPU) to conduct all our experiments.
 To run the code on CMNIST, please use the following command:
 ```
CUDA_VISIBLE_DEVICES=$GPU python -u main_adv_syn_it.py \
--trails 10 \
--dataset cmnist \
--emb_dim 300 \
--epochs 100 \
--cau_gamma 0.5 \
--adv_gamma_node 1.0 \
--adv_gamma_edge 0.8 \
--adv_dis 0.2 \
--adv_reg 0.5 \
--cau_reg 1.0 \
--causaler_lr 0.001 \
--attacker_lr 0.005 \
--test_epoch 10 --data_dir $DATA_DIR

```
 

 To run the code on Molbbbp, please use the following command:
 ```
CUDA_VISIBLE_DEVICES=$GPU python -u main_adv_mol_it.py \
--trails 10 \
--domain scaffold \
--dataset ogbg-molbbbp \
--epochs 100 \
--emb_dim 64 \
--cau_gamma 0.5 \
--adv_dis 0.5 \
--adv_reg 0.5 \
--cau_reg 0.5 \
--causaler_lr 0.001 \
--attacker_lr 0.001 --data_dir $DATA_DIR
```

To run the code on Motif, please use the following command:
 ```
CUDA_VISIBLE_DEVICES=$GPU python -u main_adv_syn_it.py \
--trails 10 \
--domain basis \
--dataset motif \
--epochs 100 \
--cau_gamma 0.5 \
--adv_gamma 1.0 \
--adv_gamma_edge 0.8 \
--adv_dis 0.2 \
--adv_reg 0.5 \
--cau_reg 1.0 \
--causaler_lr 0.001 \
--attacker_lr 0.005 \
--data_dir $DATA_DIR
```

 To run the code on Molhiv, please use the following command:
 ```
CUDA_VISIBLE_DEVICES=$GPU python -u main_adv_mol_it.py \
--trails 10 \
--domain size \
--dataset hiv \
--epochs 100 \
--emb_dim 128 \
--cau_gamma 0.1 \
--adv_gamma_node 1.0 \
--adv_gamma_edge 1.0 \
--adv_dis 1.5 \
--adv_reg 0.5 \
--cau_reg 0.5 \
--causaler_lr 0.01 \
--attacker_lr 0.01 \
--data_dir $DATA_DIR
```
