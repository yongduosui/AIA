# AdvCA
 codes of AdvCA
 To run the code on CMNIST, please use the following command:
 ```
 CUDA_VISIBLE_DEVICES=$1 python -u main_adv_syn_it.py --trails 10 --domain color --dataset cmnist \
--batch_size 128 \
--emb_dim 256 \
--epochs 100 \
--fro_layer 2 \
--bac_layer 2 \
--cau_layer 2 \
--att_layer 2 \
--cau_gamma 0.5 \
--adv_gamma_node 1.0 \
--adv_gamma_edge 1.0 \
--adv_dis 0.2 \
--adv_reg 0.5 \
--cau_reg 1.0 \
--causaler_lr 0.001 \
--attacker_lr 0.005 \
--test_epoch -1```
 
