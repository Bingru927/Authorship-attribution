import torch

bert_model_name = 'bert-base-cased'
dropout_rate = 0
bert_model_output_dim = 768
num_hiddens = 64
lstm_num_layers = 2
max_seq_len = 170
learning_rate = 1e-6
weight_decay = 0
batch_size = 32
num_epochs = 100
n_samples = 64 
best_model_name = 'best_bert_bayes21.pth'
sep = '<SEP>'

# train_file = 'darkdataset/darknet_authorship_verification_train_nodupe_anon.jsonl'
# test_file = 'darkdataset/darknet_authorship_verification_test_nodupe_anon.jsonl'
# val_file = 'darkdataset/darknet_authorship_verification_val_nodupe_anon.jsonl'
train_file = 'datasets/darkreddit_authorship_verification_train_nodupe_anon.jsonl'
test_file = 'datasets/darkreddit_authorship_verification_test_nodupe_anon.jsonl'
val_file = 'datasets/darkreddit_authorship_verification_val_nodupe_anon.jsonl'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')