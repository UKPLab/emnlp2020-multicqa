# Data Creation

Our recent [EMNLP-19 paper](https://www.aclweb.org/anthology/D19-1171/) shows that we can train a model with title-body pairs for duplicate question detection and answer selection. 
This folder contains scripts to create training datasets from arbitrary [StackExchange dumps](https://archive.org/download/stackexchange).

We can create new training sets (tsv format) with extended data by running: 
`python create_extended_train_data.py --help`; 
or only with title-body data by running:
`python create_train_data_standard.py --help`; 
