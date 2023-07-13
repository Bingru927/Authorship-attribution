import torch

bert_model_name = 'bert-base-cased'
dropout_rate = 0.1
bert_model_output_dim = 768
num_hiddens = 512
max_seq_len = 120
learning_rate = 1e-5
batch_size = 32
num_epochs = 500
weight_decay = 0
best_model_name = 'aibest.pth'

# train_file = 'datasets/darkreddit_authorship_attribution_train_anon.jsonl'
# test_file = 'datasets/darkreddit_authorship_attribution_test_anon.jsonl'
# val_file = 'datasets/darkreddit_authorship_attribution_val_anon.jsonl'
# train_file = 'datasets/train_augmentation.jsonl'
train_file = 'other_class_add/random_other_train.jsonl'
test_file = 'other_class_add/random_other_test.jsonl'
val_file = 'other_class_add/random_other_val.jsonl'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')