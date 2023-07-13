
import json
import time

import tqdm
import re
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier


texts_train = []
labels_train = []
texts_test = []
labels_test = []

with open('darkreddit_authorship_attribution_anon/darkreddit_authorship_attribution_train_anon.jsonl', 'r') as f:
    for line in tqdm.tqdm(f):
        line = json.loads(line)
        for key, value in line.items():
            if key == 'author':
                labels_train.append(value)
            elif key == 'comment':
                texts_train.append(value)
    f.close()

with open('darkreddit_authorship_attribution_anon/darkreddit_authorship_attribution_test_anon.jsonl', 'r') as f:
    for line in tqdm.tqdm(f):
        line = json.loads(line)
        for key, value in line.items():
            if key == 'author':
                labels_test.append(value)
            elif key == 'comment':
                texts_test.append(value)
    f.close()

X_train_pre = []
X_test_pre = []
Y_train = labels_train
Y_test = labels_test

# 数据预处理
for text in texts_train:
    text = re.sub(r'http\S+', '', text)
    # text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[@$%^&*()]', '', text)
    text = re.sub(r'\s+', ' ', text)
    X_train_pre.append(text)

for text in texts_test:
    text = re.sub(r'http\S+', '', text)
    # text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[@$%^&*()]', '', text)
    text = re.sub(r'\s+', ' ', text)
    X_test_pre.append(text)


start_time = time.time()
count_vectorizer = CountVectorizer(ngram_range=(1, 10), max_features=10000)
X_train = count_vectorizer.fit_transform(X_train_pre)
X_test = count_vectorizer.transform(X_test_pre)

tfidf_transformer = TfidfTransformer()
X_train = tfidf_transformer.fit_transform(X_train)
X_test = tfidf_transformer.transform(X_test)

Y_train = labels_train
Y_test = labels_test

print(X_train.shape[:])
print(X_test.shape[:])

model = RandomForestClassifier()
model.fit(X_train, Y_train)
Y_pred_test = model.predict(X_test)
Y_pred_train = model.predict(X_train)

acc = accuracy_score(Y_test, Y_pred_test)
end_time = time.time()
total_time = end_time - start_time
print("using time：", total_time, "s")
print("Accuracy on test:", acc)
print(classification_report(Y_test, Y_pred_test, target_names=model.classes_))

# cm = confusion_matrix(Y_test, Y_pred_test)
# fig, ax = plt.subplots()
# im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# ax.figure.colorbar(im, ax=ax)
# ax.set(xticks=np.arange(cm.shape[1]),
#        yticks=np.arange(cm.shape[0]),
#        xticklabels=model.classes_,
#        yticklabels=model.classes_,
#        title='CountVectorizer+TfidfTransformer+randomForest',
#        ylabel='True Label',
#        xlabel='Predicted Label')
#
# thresh = cm.max() / 2.
# for i in range(cm.shape[0]):
#     for j in range(cm.shape[1]):
#         ax.text(j, i, format(cm[i, j], 'd'),
#                 ha="center", va="center",
#                 color="white" if cm[i, j] > thresh else "black")
#
# # 调整图表布局
# fig.tight_layout()
# plt.show()


print("------------------------------------------------------")
print("countvector")
start_time = time.time()
vectorizer3 = CountVectorizer(ngram_range=(1, 2), max_features=5000)
vectorizer3.fit(X_train_pre)
vectorizer3.fit(X_test_pre)
X_train = vectorizer3.fit_transform(X_train_pre)
X_test = vectorizer3.transform(X_test_pre)


model = RandomForestClassifier()
model.fit(X_train, Y_train)
Y_pred_test = model.predict(X_test)
# Y_pred_train = model.predict(X_train)

acc = accuracy_score(Y_test, Y_pred_test)
# acc2 = accuracy_score(Y_train, Y_pred_train)
end_time = time.time()
total_time = end_time - start_time
print("using time：", total_time, "s")
print("Accuracy on test:", acc)
print(classification_report(Y_test, Y_pred_test, target_names=model.classes_))
# print("Accuracy on train:", acc2)
# print(classification_report(Y_train, Y_pred_train, target_names=model.classes_))

# cm = confusion_matrix(Y_test, Y_pred_test)
# fig, ax = plt.subplots()
# im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# ax.figure.colorbar(im, ax=ax)
# ax.set(xticks=np.arange(cm.shape[1]),
#        yticks=np.arange(cm.shape[0]),
#        xticklabels=model.classes_,
#        yticklabels=model.classes_,
#        title='CountVectorizer+randomForest',
#        ylabel='True Label',
#        xlabel='Predicted Label')
#

# thresh = cm.max() / 2.
# for i in range(cm.shape[0]):
#     for j in range(cm.shape[1]):
#         ax.text(j, i, format(cm[i, j], 'd'),
#                 ha="center", va="center",
#                 color="white" if cm[i, j] > thresh else "black")

# fig.tight_layout()
# plt.show()
