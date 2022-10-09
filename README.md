# AdvCA
We provide a detailed code for AdvCA.
## Preparations
Please download the graph OOD datasets and OGB datasets as described in the original paper. Then modify the datapath by specifying ```--data_dir```.


## Commands
 To run the code on CMNIST, please use the following command:
 ```
CUDA_VISIBLE_DEVICES=$GPU python -u main_adv_syn_it.py \
--trails 1 \
--dataset cmnist \
--batch_size 512 \
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
--lr_scheduler None \
--test_epoch 10 --data_dir $DATA_DIR

```
 

 To run the code on Molbbbp, please use the following command:
 ```
CUDA_VISIBLE_DEVICES=$GPU python -u main_adv_mol_it.py \
--trails 10 \
--domain scaffold \
--dataset ogbg-molbbbp \
--batch_size 32 \
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
--batch_size 512 \
--epochs 100 \
--test_epoch 80 \
--cau_gamma 0.5 \
--adv_gamma 1.0 \
--adv_gamma_edge 0.8 \
--adv_dis 0.2 \
--adv_reg 0.5 \
--cau_reg 1.0 \
--causaler_lr 0.001 \
--attacker_lr 0.005 \
--lr_scheduler cos --data_dir $DATA_DIR
```

 To run the code on Molhiv, please use the following command:
 ```
CUDA_VISIBLE_DEVICES=$GPU python -u main_adv_mol_it.py \
--trails 10 \
--domain size \
--dataset hiv \
--batch_size 512 \
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
--lr_scheduler None --data_dir $DATA_DIR
```
