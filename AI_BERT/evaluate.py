from tqdm import tqdm
from configs import *
from torch import optim
from models import BertCls
from dataset import ClsDataset
from transformers import BertTokenizer
from preprocess import get_tag2idx, load_comments_and_tags, preprocess_data
from torch.utils.data import DataLoader
from metric import *
import numpy as np
import torch


def accuracy(model, dataloader):
    model.eval()
    num_acc, num_samples = 0, 0
    for input_ids, attention_mask, tags in dataloader:
        input_ids, attention_mask, tags = input_ids.to(device), attention_mask.to(device), tags.to(device)
        output = model(input_ids, attention_mask)
        num_acc += torch.sum(output.argmax(dim=-1) == tags)
        num_samples += len(input_ids)
    return float(num_acc / num_samples)


def predict(model, dataloader):
    model.eval()
    true_y = np.asarray(())
    pred_y = np.asarray(())
    for input_ids, attention_mask, tags in dataloader:
        input_ids, attention_mask, tags = input_ids.to(device), attention_mask.to(device), tags.to(device)
        output = model(input_ids, attention_mask)
        if pred_y.size ==0:
            pred_y = output.argmax(dim=-1).numpy()
            true_y = tags.numpy()
        else:
            pred_y = np.concatenate((pred_y,output.argmax(dim=-1).numpy()),axis=0)
            true_y = np.concatenate((true_y,tags.numpy()),axis=0)

    return  true_y, pred_y
    


def test_model():
    tag2idx, idx2tag = get_tag2idx(train_file)
    num_classes = len(tag2idx)
    model = BertCls(num_classes)
    model.load_state_dict(torch.load(f'bert_AI/outputs/{best_model_name}'))
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    test_dataset = ClsDataset(preprocess_data(test_file, tokenizer, max_seq_len))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    
    true_y, pred_y = predict(model, test_dataloader)
    results = evaluate_all(true_y, pred_y)

    
    print(results)


test_model()