import json
import re

import nltk
import tqdm
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import time
from scipy.stats import randint as sp_randint


texts_train = []
labels_train = []
texts_test = []
labels_test = []
print("Loading data")
with open(
        'other_class_add/random_other_train.jsonl',
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
        'other_class_add/random_other_test.jsonl',
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


for text in tqdm.tqdm(texts_train, desc="Processing train data "):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'@$%^&*()\\', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', 'newline', text)
    text = re.sub(r"[:;=][)D]|[(][=:;]", 'emoji', text)

    X_train_pre.append(text)

for text in tqdm.tqdm(texts_test, desc="Processing test data "):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'@$%^&*()\\', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', 'newline', text)
    text = re.sub(r'[:;=][)D]|[(][=:;]', 'emoji', text)

    X_test_pre.append(text)

X_train_POS = []
X_test_POS = []


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
start_time = time.time()
count_vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=10000)
X_train = count_vectorizer.fit_transform(X_train_POS)
X_test = count_vectorizer.transform(X_test_POS)
print("CountVectorizer done")
# vectorizer3 = CountVectorizer(ngram_range=(1, 2), max_features=5000)
# vectorizer3.fit(X_train_pre)
# vectorizer3.fit(X_test_pre)
# X_train = vectorizer3.fit_transform(X_train_pre)
# X_test = vectorizer3.transform(X_test_pre)

print("Start TfidfTransformer")
tfidf_transformer = TfidfTransformer()
X_train = tfidf_transformer.fit_transform(X_train)
X_test = tfidf_transformer.transform(X_test)
print("TfidfTransformer done")

Y_train = labels_train
Y_test = labels_test
# print(X_train.shape[:])
# print(X_test.shape[:])

print("Start train mlp")
# model = SVC(max_iter=10000, C=2, kernel='linear', decision_function_shape='ovr')
model = MLPClassifier(max_iter=100000, solver='adam', learning_rate='invscaling', hidden_layer_sizes=(172,),
                      alpha=1e-05, activation='logistic')
model.fit(X_train, Y_train)
print("Train finished")

print("Test model")
Y_pred_test = model.predict(X_test)
# Y_pred_train = model.predict(X_train)

acc = accuracy_score(Y_test, Y_pred_test)
# # acc2 = accuracy_score(Y_train, Y_pred_train)
#
end_time = time.time()
total_time = end_time - start_time
print("using time：", total_time, "s")
print("Accuracy on test:", acc)
print(classification_report(Y_test, Y_pred_test, target_names=model.classes_))
# print("Accuracy on train:", acc2)
# print(classification_report(Y_train, Y_pred_train, target_names=model.classes_))

# Charting the confusion matrix
cm = confusion_matrix(Y_test, Y_pred_test)
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=model.classes_,
       yticklabels=model.classes_,
       title='MLPClassifier',
       ylabel='True Label',
       xlabel='Predicted Label')
ax.set_xticklabels(model.classes_, rotation=45)

# Label the values in the matrix grid
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

# Adjusting the chart layout
fig.tight_layout()
plt.savefig('metricmlp.png', dpi=300, bbox_inches='tight')
plt.show()

# RandomizedSearchCV
param_dist = {
    'hidden_layer_sizes': [(sp_randint.rvs(0, 200),) for _ in range(10)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'alpha': np.logspace(-5, -1, 5),
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'max_iter': [100000],
}

mlp = MLPClassifier()

random_search = RandomizedSearchCV(mlp, param_distributions=param_dist, n_iter=50, cv=5, n_jobs=-1)

random_search.fit(X_train, Y_train)

print("Best parameters: ", random_search.best_params_)
print("Best score: ", random_search.best_score_)

# Extract the results from the RandomizedSearchCV object
results = random_search.cv_results_

# Extract the alpha values and their corresponding mean test scores
alphas = results['param_alpha']
mean_scores = results['mean_test_score']

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(np.arange(len(alphas)), mean_scores, tick_label=alphas)
plt.xlabel('Alpha')
plt.ylabel('Mean Test Score')
plt.title('RandomizedSearchCV Results for MLPClassifier (max_iter=100000)')

# Highlight the best alpha
best_alpha = random_search.best_params_['alpha']
best_index = list(alphas).index(best_alpha)
plt.bar(best_index, mean_scores[best_index], color='red', label='Best Alpha')
plt.legend()

plt.show()
plt.savefig('mlpgyper.png', dpi=300, bbox_inches='tight')

