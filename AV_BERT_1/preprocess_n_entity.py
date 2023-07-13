import json
import torch
from configs import *
from transformers import BertTokenizer
import re
import spacy
# spacy.require_gpu()
nlp = spacy.load('en_core_web_md')

def replace_named_entities(text: str) -> str:
    nlp = spacy.load('en_core_web_md')
    entlist = []
    doc = nlp(text)
    for i,ent in enumerate(doc.ents):
        #print('named entity {}: {}, label {}'.format(i,ent.text,ent.label_))
        entlist.append((ent.text,ent.label_))
    
    for str_, label_ in entlist:
        text = text.replace(str_, label_)
    #print(text)
    return text


def read_jsonl(file_name):
    items = open(file_name, 'r', encoding='utf-8').read().splitlines()
    items = [json.loads(item) for item in items]
    return items


def get_tag2idx(file_name):

    #tag2idx = {True: 1 ,False:0}
    #idx2tag = {1:True, 0:False}
    
    items = read_jsonl(file_name)
    tag_counter = {}
    for item in items:
        tag_counter[item['same']] = tag_counter.get(item['same'], 0) + 1
    tag2idx = {tag: idx for idx, tag in enumerate(list(tag_counter.keys()))}
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    
    return tag2idx, idx2tag

def process(texts):
    # text = re.sub(r'http\S+', 'URL', texts)  # Remove website link
    # # text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    # text = re.sub(r'[@$%^&*()]', '', text)  # Remove illegal characters
    # text = re.sub(r'\s+', ' ', text)
    # text = re.sub(r'[0-9]','',text)
    text = replace_named_entities(texts) #replace named entitis
    return text

def load_comments_and_tags(tag2idx, file_name):
    items = read_jsonl(file_name)
    comments, tags = [], []
    for item in items:
        s1 = process(item['pair'][0])
        s2 = process(item['pair'][1])
        pair =s1 +sep+s2
        comments.append(pair)
        tags.append(tag2idx[item['same']])
    return comments, tags


def preprocess_data(file_name, tokenizer, max_seq_len):
    input_ids, attention_masks, label_ids = [], [], []
    tag2idx, idx2tag = get_tag2idx(file_name)
    comments, tags = load_comments_and_tags(tag2idx, file_name)
    for comment, tag in zip(comments, tags):
        encoded_dict = tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=max_seq_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        label_ids.append(tag)
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    label_ids = torch.tensor(label_ids, dtype=torch.long)
    return input_ids, attention_masks, label_ids


if __name__ == '__main__':
    input_ids, attention_masks, label_ids = preprocess_data(train_file, BertTokenizer.from_pretrained(bert_model_name), max_seq_len)
    print(input_ids.shape)
    print(attention_masks.shape)
    print(label_ids.shape)