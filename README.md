# README

## Code Organization

### Training Scripts

`citation.py `: for three medium-sized datasets, i.e., Cora, Citeseer, and Pubmed

`hetero.py`: for two heterogeneous datasets, i.e., ACM and DBLP

`ogbn.py`: for the Ogbn-Products dataset

You can refer to  `run.sh`  for specific running commands.

For example, if you want to test the performance of RWGSL with GCN on the Cora dataset, you can run the command as follows:

`python citation.py --dataset cora --model GCN --lr 0.1 --hidden_channels 256 --dropout 0.3 --update 5 --walk_len 10 --high 0.95 --low 0.35 --lp_num_layers 2 --lp_alpha 0.4 --first_coe 0.5 --second_coe 0.6 --third_coe 1.4`

Particularly, if you need to test the performance of GCN on the original Cora, you can append `--train_ori` to the above command.

### Functional Codes

`sample.py`: neighborhood sampling, similarity calculation, random walk, and structure modification

`metrics.py`: accuracy and $F_1$ score calculation

`models.py`: the definitions of neural network classes

`normalization.py`: different kinds of matrix normalization functions

`utils.py`: utility functions for data preparation, model instantiation, etc.

### Data

All data should be placed into the `./data` directory. 
