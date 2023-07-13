import os
import json
import pickle
from itertools import chain

import pandas as pd
import numpy as np
import re
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from nltk.tokenize import word_tokenize
import statistics
from gensim.models import Word2Vec
import gensim.downloader as api


pre_models = list(api.info()['models'].keys())
print(pre_models)

# help function
def get_data_path(data_dir):
    dirs = os.listdir(data_dir)

    for path in dirs:
        if "_train_" in path:
            train_path = os.path.join(data_dir, path)
        elif "_val_" in path:
            val_path = os.path.join(data_dir, path)
        elif "_test_" in path:
            test_path = os.path.join(data_dir, path)
    return train_path, val_path, test_path

def getDataJSON(route):
    with open(route,"r",encoding="utf-8") as f:
        result = [json.loads(line) for line in f.read().splitlines()]
    return result

def preprocess(text):
    text = re.sub(r'http\S+', 'URL', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[@$%^&*()]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return word_tokenize(text)

def get_data(path):
    data = pd.DataFrame(getDataJSON(path))
    data['Text_Tokenized'] = data['comment'].str.lower().apply(preprocess)
    return data


def w2v_model(pre_model, data_extra):
    """
    pre_model : str, 可用预训练的模型之一
    data_extra: list_of_tokens 用来微调，一般为train，val，test中的文本token

    return: 微调好的word2vector 模型

    """
    # model = Word2Vec.load("word_embedding_128")
    model = api.load(pre_model)
    vector_size_n = int(pre_model.split('-')[-1])
    print(vector_size_n)
    model_2 = Word2Vec(
        vector_size=vector_size_n,
        window=100,
        min_count=1,
        sg=0,  # 0=CBOW, 1=Skip-gram
        epochs=5)
    model_2.build_vocab(data_extra)
    # model_2.save('word_embedding_train')
    model.add_vectors(model_2.wv.index_to_key, model_2.wv.vectors)
    return model


def get_features(model, df):
    """
    model: word2vector model
    df: generally the return value of get_data, can be train_df, val_df, test_df
    return the sorted vectors, which can be used only for training

    """
    words = set(model.index_to_key)
    df['Text_vect'] = np.array([np.array([model[i] for i in ls if i in words])
                                for ls in df['Text_Tokenized']], dtype=object)
    text_vect_avg = []
    for v in df['Text_vect']:
        if v.size:
            text_vect_avg.append(v.mean(axis=0))
        else:
            print("1")
            text_vect_avg.append(np.zeros(1000, dtype=float))

    df['Text_vect_avg'] = text_vect_avg
    return text_vect_avg

data_dir = "darkreddit_authorship_attribution_anon"
pre_model = 'glove-twitter-25'

train_path, val_path, test_path = get_data_path(data_dir)
train_df = get_data(train_path)
val_df = get_data(val_path)
test_df = get_data(test_path)

# all_text = list(chain(train_df['Text_Tokenized'], test_df['Text_Tokenized'], val_df['Text_Tokenized']))

data_extra = pd.concat([train_df['Text_Tokenized'], test_df['Text_Tokenized'], val_df['Text_Tokenized']], axis=0)
# data_extra = all_text
# print(data_extra)
w2v = w2v_model(pre_model, data_extra)
# w2v = pickle.load(open("/Users/bingru/Workspace/ML/IndividualProject/project/word2vec/word2vec.pkl", "rb"))
# print(w2v.most_similar("happy"))
w2v.save("word2vec_extra.pkl")

X_train = get_features(w2v, train_df)
y_train = train_df['author']
X_test = get_features(w2v, test_df)
y_test = test_df['author']
print(len(X_train))
print(train_df)


# model = SVC(max_iter=10000, C=2, kernel='linear', decision_function_shape='ovr')
model = MLPClassifier(max_iter=10000, hidden_layer_sizes=(175,), solver='sgd', activation='relu')
model.fit(X_train, y_train)
# model = MLPClassifier(hidden_layer_sizes=(100, 256), max_iter=1000)
# model.fit(X_train, y_train)
print("Train finished")

print("Test model")
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)


acc = accuracy_score(y_test, y_pred_test)
acc2 = accuracy_score(y_train, y_pred_train)

print("Accuracy on test:", acc)
print(classification_report(y_test, y_pred_test))
print("Accuracy on train:", acc2)
print(classification_report(y_train, y_pred_train))


