# GCN
python citation.py --dataset cora --model GCN --lr 0.1 --hidden_channels 256 --dropout 0.3 --update 5 --walk_len 10 --high 0.95 --low 0.35 --lp_num_layers 2 --lp_alpha 0.4 --first_coe 0.5 --second_coe 0.6 --third_coe 1.4 # 0.8431
python citation.py --dataset cora --model SAGE --lr 0.01 --hidden_channels 128 --dropout 0.3 --weight_decay 1e-3 
python citation.py --dataset cora --model GAT --lr 5e-3 --hidden_channels 128 --dropout 0.6 --weight_decay 5e-3
python citation.py --dataset citeseer --model GCN --lr 0.05 --weight_decay 5e-4 --hidden_channels 64 --dropout 0.2 --update 6 --walk_len 17 --high 0.8 --low 0.4 --lp_num_layers 3 --lp_alpha 0.3 --first_coe 0.4 --second_coe 0.9 --third_coe 1.1 # 0.7371
python citation.py --dataset citeseer --model SAGE --lr 0.01 --hidden_channels 128 --dropout 0.2 --weight_decay 1e-3
python citation.py --dataset citeseer --model GAT --lr 1e-3 --hidden_channels 256 --dropout 0.6 --weight_decay 5e-4
python citation.py --dataset pubmed --model GCN --lr 0.1 --weight_decay 5e-4 --hidden_channels 256 --dropout 0.1 --update 5 --walk_len 12 --high 1.25 --low 0.35 --lp_num_layers 2 --lp_alpha 0.4 --first_coe 0.5 --second_coe 0.1 --third_coe 1.6 # 0.8026
python citation.py --dataset pubmed --model SAGE --lr 0.01 --weight_decay 1e-3 --hidden_channels 256 --dropout 0.2
python citation.py --dataset pubmed --model GAT --lr 1e-3 --hidden_channels 256 --dropout 0.6 --weight_decay 5e-4 --nheads 8,8

# PRODUCTS
python ogbn.py --dataset ogbn_products --model SIGN --model_degree 5 --input_drop 0.3 --dropout 0.4 --lr 0.001 --update 2 --walk_len 5 --high 0.95 --low 0.45 --lp_num_layers 4 --lp_alpha 0.5 --first_coe 0.8 --second_coe 1.3 --third_coe 0.5
python ogbn.py --dataset ogbn_products --model SIGN --model_degree 5 --input_drop 0.3 --dropout 0.4 --lr 0.001 --noise delete --update 3 --walk_len 7 --high 1.0 --low 0.3 --lp_num_layers 4 --lp_alpha 0.5 --first_coe 0.7 --second_coe 0.5 --third_coe 1.2
python ogbn.py --dataset ogbn_products --model SGC --model_degree 5 --lr 0.001  --epochs 1000 
python ogbn.py --dataset ogbn_products --model SAGE --num_layers 3

# ACM 
python hetero.py --dataset acm --update 4 --walk_len 5 --high 1.0 --low 0.45 --lp_num_layers 3 --lp_alpha 0.7 --first_coe 0.4 --second_coe 1.4 --third_coe 2.0 # macro-F1 0.9365
# DBLP
python hetero.py --dataset dblp --update 6 --walk_len 13 --high 1.2 --low 0.4 --lp_num_layers 2 --lp_alpha 0.8 --first_coe 0.7 --second_coe 1.8 --third_coe 1.5 # macro-F1 0.9213