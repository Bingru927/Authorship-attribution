import json
import re

import nltk
import tqdm
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

# 从JSON文件中读取文本和标签
texts_train = []
labels_train = []
texts_test = []
labels_test = []
print("Loading data")
with open(
        'darkreddit_authorship_attribution_train_anon.jsonl',
        'r') as f:
    for line in f:
        line = json.loads(line)
        for key, value in line.items():
            if key == 'author':
                labels_train.append(value)
            elif key == 'comment':
                texts_train.append(value)
    f.close()
with open('train_augmentation9_10.jsonl', 'r') as f:
    for line in f:
        line = json.loads(line)
        for key, value in line.items():
            if key == 'author':
                labels_train.append(value)
            elif key == 'comment':
                texts_train.append(value)
    f.close()
with open('darkreddit_authorship_attribution_test_anon.jsonl',
          'r') as f:
    for line in f:
        line = json.loads(line)
        for key, value in line.items():
            if key == 'author':
                labels_test.append(value)
            elif key == 'comment':
                texts_test.append(value)
    f.close()
print("Loading finished")

# print(len(texts_train))
# print(len(labels_train))
# print(len(texts_test))
# print(len(labels_test))

X_train_pre = []
X_test_pre = []
Y_train = labels_train
Y_test = labels_test

# 数据预处理
for text in tqdm.tqdm(texts_train, desc="Processing train data "):
    text = re.sub(r'http\S+', 'URL', text)
    # text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[@$%^&*()]', '', text)
    text = re.sub(r'\s+', ' ', text)
    X_train_pre.append(text)

for text in tqdm.tqdm(texts_test, desc="Processing test data "):
    text = re.sub(r'http\S+', 'URL', text)
    # text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[@$%^&*()]', '', text)
    text = re.sub(r'\s+', ' ', text)
    X_test_pre.append(text)

X_train_POS = []
X_test_POS = []

# 词性标注
for text in tqdm.tqdm(X_train_pre, desc="Make train POS "):
    t = nltk.word_tokenize(text)
    t = nltk.pos_tag(t)
    tag = []
    for i in t:
        tag.append(nltk.tag.util.tuple2str(i))
    text = " ".join(tag)
    X_train_POS.append(text)

for text in tqdm.tqdm(X_test_pre, desc="Make test POS "):
    t = nltk.word_tokenize(text)
    t = nltk.pos_tag(t)
    tag = []
    for i in t:
        tag.append(nltk.tag.util.tuple2str(i))
    text = " ".join(tag)
    X_test_POS.append(text)


print("Start CountVectorizer")
count_vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=10000)
X_train = count_vectorizer.fit_transform(X_train_POS)
X_test = count_vectorizer.transform(X_test_POS)

print("Start TfidfTransformer")

tfidf_transformer = TfidfTransformer()
X_train = tfidf_transformer.fit_transform(X_train)
X_test = tfidf_transformer.transform(X_test)
print("TfidfTransformer done")

Y_train = labels_train
Y_test = labels_test
label_map = {'user1': 0, 'user2': 1, 'user3': 2, 'user4': 3, 'user5': 4, 'user6': 5, 'user7': 6, 'user8': 7, 'user9': 8,
             'user10': 9}
Y_train = [label_map[label] for label in Y_train]
Y_test = [label_map[label] for label in Y_test]
# n = np.arange(1, 100)
# param_grid = {'n_estimators': n}
tree = DecisionTreeClassifier(max_depth=1000, min_samples_split=10, min_samples_leaf=10, max_features=10000)
ada = AdaBoostClassifier(base_estimator=tree,algorithm='SAMME.R',n_estimators=50)
ada.fit(X_train, Y_train)
# a1 = ada.score(x_train, y_train)
# a2 = ada.score(x_test, y_test)
print(f"Test set accuracy: {ada.score(X_test, Y_test)}")
print("Test model")
Y_pred_test = ada.predict(X_test)
acc = accuracy_score(Y_test, Y_pred_test)
print("Accuracy on test:", acc)
print(classification_report(Y_test, Y_pred_test))
# AdaBoostRegressor_GridSearch = GridSearchCV(estimator=ada, param_grid=param_grid, cv=3, return_train_score=True)
# AdaBoostRegressor_GridSearch.fit(X_train, Y_train)
# AdaBoostRegressor_GridSearch.best_params_
# tree = DecisionTreeRegressor()

# min_samples_leaf = [5, 10, 15, 20, 30, 40, 50, 70, 100]
# param_grid = {'max_depth': min_samples_leaf}
# dt_GridSearch = GridSearchCV(estimator=tree, param_grid=param_grid, cv=5, return_train_score=True)
# dt_GridSearch.fit(X_train, Y_train)
# print(dt_GridSearch.best_params_)
# plt.plot([5, 10, 15, 20, 30, 40, 50, 70, 100], dt_GridSearch.cv_results_["mean_train_score"], "ko-", label='train')
# plt.plot([5, 10, 15, 20, 30, 40, 50, 70, 100], dt_GridSearch.cv_results_["mean_test_score"], "g*-", label='test')

# plt.xlabel('min_samples_leaf', fontsize=12)
# plt.ylabel('mean score', fontsize=12)
# plt.title('Hyperparameter Tuning', pad=15, size=15)
# plt.title('Hyperparameter tuning of min_samples_leaf', pad=15, size=15)
# plt.legend(loc='best', labelspacing=2, handlelength=3, fontsize=10, shadow=True)
# plt.show()
