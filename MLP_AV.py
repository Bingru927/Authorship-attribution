import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import json 
from pandas.io.json import json_normalize
from sklearn.svm import SVC
#from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import os
from sklearn.feature_extraction.text import CountVectorizer
import json

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfTransformer
import json
import re

import nltk
import tqdm
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import time

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

data_dir = "darkreddit_authorship_verification_anon"
def get_data_path(data_dir):
    dirs = os.listdir(data_dir)
    for path in dirs:
        if "_train_" in path:
            train_path = os.path.join(data_dir,path)
        elif "_val_" in path:
            val_path = os.path.join(data_dir,path)
        elif "_test_" in path:
            test_path = os.path.join(data_dir,path)
    return train_path, val_path, test_path

def getDataJSON(route):
    with open(route,"r",encoding="utf-8") as f:
        result = [json.loads(line) for line in f.read().splitlines()]
    return result

def get_data(path):
    data = pd.DataFrame(getDataJSON(path)).set_index("id")
    data[["text1","text2"]] = pd.DataFrame(data.pair.tolist(), index= data.index)
    del data["pair"]
    return data

train_path, val_path, test_path = get_data_path(data_dir)

train = get_data(train_path)
test = get_data(test_path)


X_train_pre1 = []
X_train_pre2 = []

X_test_pre1 = []
X_test_pre2 = []

# Y_train = labels_train
# Y_test = labels_test

# 数据预处理

for text in tqdm.tqdm(train["text1"], desc="Processing train data "):
    text = re.sub(r'http\S+', '', text)  # 移除网站链接
    text = re.sub(r'[^\w\s]', '', text)  # 移除标点符号
    text = re.sub(r'@$%^&*()\\', '', text)  # 移除非法字符
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', 'newline', text)
    text = re.sub(r"[:;=][)D]|[(][=:;]", 'emoji', text)

    X_train_pre1.append(text)

for text in tqdm.tqdm(train["text2"], desc="Processing test data "):
    text = re.sub(r'http\S+', '', text)  # 移除网站链接
    text = re.sub(r'[^\w\s]', '', text)  # 移除标点符号
    text = re.sub(r'@$%^&*()\\', '', text)  # 移除非法字符
    text = re.sub(r'\s+', ' ', text)  # 替换双空格
    text = re.sub(r'\n+', 'newline', text)
    text = re.sub(r'[:;=][)D]|[(][=:;]', 'emoji', text)

    X_train_pre2.append(text)

for text in tqdm.tqdm(test["text1"], desc="Processing train data "):
    text = re.sub(r'http\S+', '', text)  # 移除网站链接
    text = re.sub(r'[^\w\s]', '', text)  # 移除标点符号
    text = re.sub(r'@$%^&*()\\', '', text)  # 移除非法字符
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', 'newline', text)
    text = re.sub(r"[:;=][)D]|[(][=:;]", 'emoji', text)

    X_test_pre1.append(text)

for text in tqdm.tqdm(test["text2"], desc="Processing test data "):
    text = re.sub(r'http\S+', '', text)  # 移除网站链接
    text = re.sub(r'[^\w\s]', '', text)  # 移除标点符号
    text = re.sub(r'@$%^&*()\\', '', text)  # 移除非法字符
    text = re.sub(r'\s+', ' ', text)  # 替换双空格
    text = re.sub(r'\n+', 'newline', text)
    text = re.sub(r'[:;=][)D]|[(][=:;]', 'emoji', text)

    X_test_pre2.append(text)
X_train_POS1 = X_train_pre1
X_train_POS2 = X_train_pre1

X_test_POS1 = X_test_pre1
X_test_POS2 = X_test_pre2
# for text in tqdm.tqdm(X_train_pre1, desc="Make train POS "):
#     t = nltk.word_tokenize(text)
#     t = nltk.pos_tag(t)
#     tag = []
#     for i in t:
#         tag.append(nltk.tag.util.tuple2str(i))
#     text = " ".join(tag)
#     X_train_POS1.append(text)

# for text in tqdm.tqdm(X_train_pre2, desc="Make train POS "):
#     t = nltk.word_tokenize(text)
#     t = nltk.pos_tag(t)
#     tag = []
#     for i in t:
#         tag.append(nltk.tag.util.tuple2str(i))
#     text = " ".join(tag)
#     X_train_POS2.append(text)

# for text in tqdm.tqdm(X_test_pre1, desc="Make test POS "):
#     t = nltk.word_tokenize(text)
#     t = nltk.pos_tag(t)
#     tag = []
#     for i in t:
#         tag.append(nltk.tag.util.tuple2str(i))
#     text = " ".join(tag)
#     X_test_POS1.append(text)

# for text in tqdm.tqdm(X_test_pre2, desc="Make test POS "):
#     t = nltk.word_tokenize(text)
#     t = nltk.pos_tag(t)
#     tag = []
#     for i in t:
#         tag.append(nltk.tag.util.tuple2str(i))
#     text = " ".join(tag)
#     X_test_POS2.append(text)

batch_size = 1000  
num_batches_train1 = int(np.ceil(len(X_train_POS1) / float(batch_size)))  
num_batches_test1 = int(np.ceil(len(X_test_POS1) / float(batch_size)))  

vectorizer = TfidfVectorizer(max_features=10000)
# X_train = []
# X_test = []
model = MLPClassifier(max_iter=100000, solver='adam', learning_rate='invscaling', hidden_layer_sizes=(172,),
                      alpha=1e-05, activation='logistic')
Y_train = train['same'].astype(int).values
Y_test = test['same'].astype(int).values
for i in range(num_batches_train1):
    start = i * batch_size
    end = min((i + 1) * batch_size, len(X_train_POS1))
    batch_train1 = X_train_POS1[start:end]
    batch_train2 = X_train_POS2[start:end]
    batch_tfidf_train = vectorizer.fit(batch_train1).fit(batch_train2)
    x_train= vectorizer.transform(batch_train1).toarray()- vectorizer.transform(batch_train2).toarray()
    print("Start train mlp")
    y_train = Y_train[start:end]
    model.fit(x_train, y_train)
    print("Train finished")
# for i in range(num_batches_test1):
#     start = i * batch_size
#     end = min((i + 1) * batch_size, len(X_test_POS1))
#     batch_test1 = X_test_POS1[start:end]
#     batch_test2 = X_test_POS2[start:end]
#     x_test= vectorizer.transform(batch_test1).toarray()- vectorizer.transform(batch_test2).toarray()
#     X_test.append(x_test)
Y_test = test['same'].astype(int).values

print("Test model")
X_test= vectorizer.transform(X_test_POS1).toarray() - vectorizer.transform(X_test_POS2).toarray()
Y_pred_test = model.predict(X_test)

acc = accuracy_score(Y_test, Y_pred_test)
print("Accuracy on test:", acc)
print(classification_report(Y_test, Y_pred_test))