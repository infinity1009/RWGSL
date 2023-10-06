# GCN
CUDA_VISIBLE_DEVICES=0 python citation.py --noise none --dataset cora --model GCN --lr 0.1 --hidden_channels 256 --dropout 0.3
CUDA_VISIBLE_DEVICES=0 python citation.py --noise none --dataset citeseer --model GCN --lr 0.05 --weight_decay 5e-4 --hidden_channels 64 --dropout 0.2 
CUDA_VISIBLE_DEVICES=0 python citation.py --noise none --dataset pubmed --model GCN --lr 0.1 --weight_decay 0.0005 --hidden_channels 256 --dropout 0.1

# REDDIT
CUDA_VISIBLE_DEVICES=0 python reddit.py --noise none --train_ori --model SGC --epochs 15
CUDA_VISIBLE_DEVICES=0 python reddit.py --noise none --lr 0.14 --weight_decay 1e-4 --hidden_channels 128 --dropout 0 --model SSGC --epochs 15

# PRODUCTS
CUDA_VISIBLE_DEVICES=0 python ogbn.py --dataset ogbn_products --model SGC --epochs 1000 --model_degree 5 --noise none --lr 0.001
CUDA_VISIBLE_DEVICES=0 python ogbn.py --dataset ogbn_products --model SIGN --epochs 500 --model_degree 3 --hidden_channels 256 --noise none --lr 0.01
