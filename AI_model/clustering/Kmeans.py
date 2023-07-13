import json
import re

import numpy as np
import matplotlib.pyplot as plt
import tqdm
from matplotlib.colors import LogNorm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
import imageio.v3 as iio

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
# with open('data/train_augmentation9_10.jsonl', 'r') as f:
#     for line in f:
#         line = json.loads(line)
#         for key, value in line.items():
#             if key == 'author':
#                 labels_train.append(value)
#             elif key == 'comment':
#                 texts_train.append(value)
#     f.close()
with open(
        '/Users/bingru/Workspace/ML/IndividualProject/Data/darkreddit_authorship_attribution_anon/darkreddit_authorship_attribution_test_anon.jsonl',
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

X_train = []
X_test = []
Y_train = labels_train
Y_test = labels_test

# 数据预处理
for text in tqdm.tqdm(texts_train, desc="Processing train data "):
    text = re.sub(r'http\S+', 'URL', text)
    # text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[@$%^&*()]', '', text)
    text = re.sub(r'\s+', ' ', text)
    X_train.append(text)

for text in tqdm.tqdm(texts_test, desc="Processing test data "):
    text = re.sub(r'http\S+', 'URL', text)
    # text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[@$%^&*()]', '', text)
    text = re.sub(r'\s+', ' ', text)
    X_test.append(text)

print("Start CountVectorizer")
count_vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=10000)
X_train = count_vectorizer.fit_transform(X_train)
X_test = count_vectorizer.transform(X_test)

print("Start TfidfTransformer")
tfidf_transformer = TfidfTransformer()
X_train = tfidf_transformer.fit_transform(X_train)
X_test = tfidf_transformer.transform(X_test)
print("TfidfTransformer done")

Y_train = labels_train
Y_test = labels_test

print(X_train.shape[:])
print(X_test.shape[:])

K_values = range(1, 20)
scores = []

for K in K_values:
    kmeans = KMeans(K).fit(X_train)
    scores.append(-kmeans.score(X_train))
plt.figure()
plt.plot(K_values, scores)
plt.show()

def plot_k_means(X, cluster_assignments, centroid_locations):
    plt.figure(figsize=(6, 6))
    plt.viridis() # Set colour map
    plt.scatter(X[:, 0], X[:, 1], s=20, c=cluster_assignments, alpha=0.8) # plot data points
    plt.scatter(centroid_locations[:, 0], centroid_locations[:, 1], s=200, marker='X', c=range(K), edgecolors='k') # plot centroids
    plt.show()

kmeans = KMeans(K, init='random').fit(X_train)
cluster_assignments = kmeans.predict(X_train)
centroid_locations = kmeans.cluster_centers_

plot_k_means(X_train, cluster_assignments, centroid_locations)


num_clusters = 10
km_cluster = KMeans(n_clusters=num_clusters, max_iter=300, n_init=40, init='k-means++')
result = km_cluster.fit_predict(X_train)

print("Predicting result: ", result)
