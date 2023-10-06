# README

## Code Organization

### Training Scripts

`citation.py `: for three medium-sized datasets, i.e., Cora, Citeseer, and Pubmed

`reddit.py`: for the Reddit dataset

`ogbn.py`: for the Ogbn-Products dataset

You can refer to  `run.sh`  for specific running commands.

For example, if you want to test the performance of GCN on the optimized Cora, you can run the command as follows:

`CUDA_VISIBLE_DEVICES=0 python citation.py --noise none --dataset cora --model GCN --lr 0.1 --hidden_channels 256 --dropout 0.3`

Particularly, if you need to test the performance of GCN on the original Cora, you can append `--train_ori` to the above command.

### Functional Codes

`sample.py`: neighborhood sampling, similarity calculation, random walk, and structure modification

`metrics.py`: accuracy and $F_1$ score calculation

`models.py`: the definitions of neural network classes

`normalization.py`: different kinds of matrix normalization functions

`utils.py`: utility functions for data preparation, model instantiation, etc.

### Data

All data should be placed into the `./data` directory. 

