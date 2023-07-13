import torch

bert_model_name = 'bert-base-cased'
dropout_rate = 0
bert_model_output_dim = 768
num_hiddens = 384
# max_seq_len = 170
# max_seq_len = 240
max_seq_len = 290
learning_rate = 1e-6
batch_size = 32
num_epochs = 100
weight_decay = 0
best_model_name = 'best_avall3.pth'
sep = '<SEP>'

# train_file = 'datasets/darkreddit_authorship_verification_train_nodupe_anon.jsonl'
# test_file = 'datasets/darkreddit_authorship_verification_test_nodupe_anon.jsonl'
# val_file = 'datasets/darkreddit_authorship_verification_val_nodupe_anon.jsonl'

# train_file = 'darknet_authorship_verification_silkroad1_anon/darknet_authorship_verification_train_nodupe_anon.jsonl'
# test_file = 'darknet_authorship_verification_silkroad1_anon/darknet_authorship_verification_test_nodupe_anon.jsonl'
# val_file = 'darknet_authorship_verification_silkroad1_anon/darknet_authorship_verification_val_nodupe_anon.jsonl'


# train_file = 'darknet_authorship_verification_agora_anon/darknet_authorship_verification_train_nodupe_anon.jsonl'
# test_file = 'darknet_authorship_verification_agora_anon/darknet_authorship_verification_test_nodupe_anon.jsonl'
# val_file ='darknet_authorship_verification_agora_anon/darknet_authorship_verification_val_nodupe_anon.jsonl'

# train_file = 'all/merged_train.jsonl'
# test_file = 'all/merged_test.jsonl'
# val_file = 'all/merged_val.jsonl'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')