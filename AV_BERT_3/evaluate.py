from tqdm import tqdm
from configs import *
from torch import optim
from models import BertCls
from dataset import ClsDataset
from BertLSTMBayesClassifier import BertLSTMBayesClassifier
from transformers import BertTokenizer
from preprocess import get_tag2idx, load_comments_and_tags, preprocess_data
from torch.utils.data import DataLoader
from metric import *
import numpy as np
import torch


def score(model, dataloader):
    model.eval()
    scores = []
    true_y = []
    for input_ids, attention_mask, tags in dataloader:
        input_ids, attention_mask, tags = input_ids.to(device), attention_mask.to(device), tags.to(device)
        output = model(input_ids, attention_mask)
        scores.extend(output.tolist())
        true_y.extend(tags.tolist())
        #pred_y.extend(output.argmax(dim=-1).tolist())

    return true_y,scores

def test_model():
    tag2idx, idx2tag = get_tag2idx(train_file)
    num_classes = len(tag2idx)
    model = SimpleBert(num_classes)
    model.load_state_dict(torch.load(f'outputs/{best_model_name}'))
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    test_dataset = ClsDataset(preprocess_data(test_file, tokenizer, max_seq_len))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    tags,scores= score(model, test_dataloader)
    results = evaluate_all(tags,scores)
    print(f'test acc {results}')

test_model()