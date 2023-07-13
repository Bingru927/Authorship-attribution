import json
import re
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

# Reading text and tags from JSON files
texts_train = []
labels_train = []
texts_test = []
labels_test = []
print("Loading data")
with open(
        'darkreddit_authorship_attribution_anon/darkreddit_authorship_attribution_train_anon.jsonl',
        'r') as f:
    for line in f:
        line = json.loads(line)
        for key, value in line.items():
            if key == 'author':
                labels_train.append(value)
            elif key == 'comment':
                texts_train.append(value)
    f.close()
with open(
        'darkreddit_authorship_attribution_anon/darkreddit_authorship_attribution_test_anon.jsonl',
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

X_train_pre = []
X_test_pre = []
Y_train = labels_train
Y_test = labels_test

# 数据预处理
for text in tqdm.tqdm(texts_train, desc="Processing train data "):
    text = re.sub(r'http\S+', ' ', text)
    # text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[@$%^&*()]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    X_train_pre.append(text)

for text in tqdm.tqdm(texts_test, desc="Processing test data "):
    text = re.sub(r'http\S+', ' ', text)
    # text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'[@$%^&*()]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    X_test_pre.append(text)
# Using TFIDF Vectorizer to generate word frequency vectors
total = []
time_used = []
# for i in [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]:
start_time = time.time()
# vectorizer3 = CountVectorizer(ngram_range=(1, 2), max_features=5000)
# vectorizer3.fit(X_train_pre)
# vectorizer3.fit(X_test_pre)
# X_train = vectorizer3.fit_transform(X_train_pre)
# X_test = vectorizer3.transform(X_test_pre)
count_vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=10000)
X_train = count_vectorizer.fit_transform(X_train_pre)
X_test = count_vectorizer.transform(X_test_pre)
tfidf_transformer = TfidfTransformer()
X_train = tfidf_transformer.fit_transform(X_train)
X_test = tfidf_transformer.transform(X_test)

model = SVC(max_iter=100000)
model.fit(X_train, Y_train)
Y_pred_test = model.predict(X_test)
acc = accuracy_score(Y_test, Y_pred_test)
total.append(acc)
# acc2 = accuracy_score(Y_train, Y_pred_train)
end_time = time.time()
total_time = end_time - start_time
time_used.append(total_time)
print("using time：", total_time, "s")
print("Accuracy on test:", acc)
print(classification_report(Y_test, Y_pred_test, target_names=model.classes_))


#test TFIDF/Count Vector (change max_features/ngram)
ngram = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13),
            (1, 14), (1, 15), (1, 16), (1, 17), (1, 18), (1, 19), (1, 20)]
total = []
time_used = []
for i in [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]:
    print("countvector")
    start_time = time.time()
    # vectorizer3 = CountVectorizer(ngram_range=(1, i), max_features=1000)
    # vectorizer3.fit(X_train_pre)
    # vectorizer3.fit(X_test_pre)
    # X_train = vectorizer3.fit_transform(X_train_pre)
    # X_test = vectorizer3.transform(X_test_pre)
    count_vectorizer = CountVectorizer(ngram_range=(1, 10), max_features=i)
    X_train = count_vectorizer.fit_transform(X_train_pre)
    X_test = count_vectorizer.transform(X_test_pre)
    tfidf_transformer = TfidfTransformer()
    X_train = tfidf_transformer.fit_transform(X_train)
    X_test = tfidf_transformer.transform(X_test)

    model = SVC(max_iter=10000)
    model.fit(X_train, Y_train)
    Y_pred_test = model.predict(X_test)

    acc = accuracy_score(Y_test, Y_pred_test)
    total.append(acc)
    # acc2 = accuracy_score(Y_train, Y_pred_train)
    end_time = time.time()
    total_time = end_time - start_time
    time_used.append(total_time)
    print("using time：", total_time, "s")
    print("Accuracy on test:", acc)
    print(classification_report(Y_test, Y_pred_test, target_names=model.classes_))
print("（1，10，time）:", time_used)
print("（1，10，acc）:", total)


# import matplotlib.pyplot as plt
#
# # x_labels = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13),
# #             (1, 14), (1, 15), (1, 16), (1, 17), (1, 18), (1, 19), (1, 20)]
x_labels = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
# Sample data
y1 = total  # X values from 1 to 20
x1 = list(range(len(x_labels)))  # Y values from 1 to 20

# Create a line plot
# Extract y coordinates from the tuples
# Create a line plot
plt.plot(x1, y1)
plt.xticks(x1, x_labels, rotation=45)

# Add labels and title
plt.ylabel('Avg accuracy on test', fontsize=14)
plt.xlabel('range of n_gram', fontsize=14)
plt.title('Testing on n_gram (TFIDF)', fontsize=14)

# Display the plot
plt.savefig('n_gram_TFIDF.png', dpi=300, bbox_inches='tight')
plt.show()

y2 = time_used
x2 = list(range(len(x_labels)))
plt.plot(x2, y2)
plt.xticks(x2, x_labels, rotation=45)
plt.ylabel('Time used (s)', fontsize=14)
plt.xlabel('range of n_gram', fontsize=14)
plt.title('Used time for n_gram (TFIDF) ')
plt.savefig('n_gram_time_TFIDF.png', dpi=300, bbox_inches='tight')
plt.show()

# print("Accuracy on train:", acc2)
# print(classification_report(Y_train, Y_pred_train, target_names=model.classes_))
