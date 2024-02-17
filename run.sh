# GCN
python citation.py --random_sample --dataset cora --model GCN --lr 0.1 --hidden_channels 256 --dropout 0.3
python citation.py --random_sample --dataset cora --model SAGE --lr 0.01 --hidden_channels 128 --dropout 0.3 --weight_decay 1e-3 
python citation.py --random_sample --dataset cora --model GAT --lr 5e-3 --hidden_channels 128 --dropout 0.6 --weight_decay 5e-3
python citation.py --random_sample --dataset citeseer --model GCN --lr 0.05 --weight_decay 5e-4 --hidden_channels 64 --dropout 0.2 
python citation.py --random_sample --dataset citeseer --model SAGE --lr 0.01 --hidden_channels 128 --dropout 0.2 --weight_decay 1e-3
python citation.py --random_sample --dataset citeseer --model GAT --lr 1e-3 --hidden_channels 256 --dropout 0.6 --weight_decay 5e-4
python citation.py --random_sample --dataset pubmed --model GCN --lr 0.1 --weight_decay 5e-4 --hidden_channels 256 --dropout 0.1
python citation.py --random_sample --dataset pubmed --model SAGE --lr 0.01 --weight_decay 1e-3 --hidden_channels 256 --dropout 0.2
python citation.py --random_sample --dataset pubmed --model GAT --lr 1e-3 --hidden_channels 256 --dropout 0.6 --weight_decay 5e-4 --nheads 8,8

# PRODUCTS
python ogbn.py --dataset ogbn_products --random_sample --model SIGN --model_degree 5 --input_drop 0.3 --dropout 0.4 --lr 0.001
python ogbn.py --dataset ogbn_products --random_sample --model SGC --model_degree 5 --lr 0.001  --epochs 1000 
python ogbn.py --dataset ogbn_products --random_sample --model SAGE --num_layers 3

# ACM 
python hetero.py --dataset acm 
# DBLP
python hetero.py --dataset dblp

python ogbn.py --dataset ogbn_products --random_sample --model SAGE --num_layers 3 --update 3 --walk_len 6 --high 1.0 --low 0.5 --lp_num_layers 2 --lp_alpha 0.5 --first_coe 1.0 --second_coe 0.3 --third_coe 2.0 --separate_1 5 --pool_num 10