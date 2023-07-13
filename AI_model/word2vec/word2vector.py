import os
import json
import pickle

import joblib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from nltk.tokenize import word_tokenize
from itertools import chain
from gensim.models import Word2Vec
import re
import time



# nltk.download('punkt')


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


# emojis = list(UNICODE_EMOJI.keys())


def preprocess(text):
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # text = re.sub(r'[^\w\s]', "", text)
    # text = re.sub(r'[@$%^&*()]', "", text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"\n", ' ', text)
    # text = re.sub(r"(@[A-Za-z0-9_]+)", '', text)
    # text = [word for word in text if not word in emojis]
    return word_tokenize(text)


def getDataJSON(route):
    with open(route, "r", encoding="utf-8") as f:
        result = [json.loads(line) for line in f.read().splitlines()]
    return result


def get_data(path):
    data = pd.DataFrame(getDataJSON(path))
    data['Text_Tokenized'] = data['comment'].str.lower().apply(preprocess)
    return data


def w2v(vocab,w):
    vector_size_n_w2v = 100
    w2v_model = Word2Vec(vector_size=vector_size_n_w2v,
                         window=110,
                         min_count=27,
                         sg=0,  # 0=CBOW, 1=Skip-gram
                         epochs=w,
                         hs=1)
    # all_text = list(chain(train['Text_Tokenized'],test['Text_Tokenized'],val['Text_Tokenized']))
    w2v_model.build_vocab(all_text)
    w2v_model.train(vocab,
                    total_examples=w2v_model.corpus_count,
                    epochs=5)
    print("the len:", len(w2v_model.wv.index_to_key))
    return w2v_model


def get_feature(df, w2v_model):
    words = set(w2v_model.wv.index_to_key)
    df['Text_vect'] = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                                for ls in df['Text_Tokenized']], dtype=object)
    text_vect_avg = []
    for v in df['Text_vect']:
        if v.size:
            text_vect_avg.append(v.mean(axis=0))
        else:
            text_vect_avg.append(
                np.zeros(vector_size_n, dtype=float))
    df['Text_vect_avg'] = text_vect_avg
    return text_vect_avg


print("loading path")
data_dir = "darkreddit_authorship_attribution_anon"
train_path, val_path, test_path = get_data_path(data_dir)
print("loading done")
print("loading test data")
test = get_data(test_path)
print("loading train data")
train = get_data(train_path)
print("loading val data")
val = get_data(val_path)

all_text = list(chain(train['Text_Tokenized'], test['Text_Tokenized'], val['Text_Tokenized']))
print("creating word2vector model")
total = []
time_used = []
# for i in [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]:
start_time = time.time()
w2v_model = w2v(all_text, 70)
w2v_model.save("word2vec.pkl")
# pickle.dump(w2v_model, open('word2vec/word2vec.pkl', 'wb'))
w2v_model = pickle.load(open("word2vec.pkl", "rb"))
print("word2vector model done")
print("get train feature")
x_train = get_feature(train, w2v_model)
print("get test feature")
x_test = get_feature(test, w2v_model)
y_train = train['author']
y_test = test['author']
x_test2 = get_feature(val, w2v_model)
y_test2 = val['author']
print("Start mlp")
# model = SVC(max_iter=100000)
model = MLPClassifier(max_iter=100000)
# model = RandomForestClassifier()
model.fit(x_train, y_train)
Y_pred_test = model.predict(x_test)
# Y_pred_train = model.predict(x_train)

acc = accuracy_score(y_test, Y_pred_test)
end_time = time.time()
total_time = end_time - start_time
print("using time：", total_time, "s")
# acc2 = accuracy_score(y_train, Y_pred_train)
# model.save("MLP")
time_used.append(total_time)
print("Accuracy on test:", acc)
total.append(acc)
print(classification_report(y_test, Y_pred_test, target_names=model.classes_))
#
#
# x_labels = [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
# # x_labels = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
# # Sample data
# y1 = total  # X values from 1 to 20
# x1 = list(range(len(x_labels)))  # Y values from 1 to 20
#
# # Create a line plot
# # Extract y coordinates from the tuples
# # Create a line plot
# plt.plot(x1, y1)
# plt.xticks(x1, x_labels, rotation=45)
#
# # Add labels and title
# plt.ylabel('Avg accuracy on test', fontsize=14)
# plt.xlabel('range of epoch', fontsize=14)
# plt.title('Testing on different epochs (Word2Vec)', fontsize=14)
#
# # Display the plot
# plt.savefig('epochs_w2v.png', dpi=300, bbox_inches='tight')
# plt.show()
#
# y2 = time_used
# x2 = list(range(len(x_labels)))
# plt.plot(x2, y2)
# plt.xticks(x2, x_labels, rotation=45)
# plt.ylabel('Time used (s)', fontsize=14)
# plt.xlabel('range of epoch', fontsize=14)
# plt.title('Used time for different epochs (Word2Vec) ')
# plt.savefig('epochs_time_w2v.png', dpi=300, bbox_inches='tight')
# plt.show()

# print("Start train test")
# model = MLPClassifier(max_iter=10000,hidden_layer_sizes=(100,), solver='sgd')
# model.fit(x_train, y_train)
# print("Train finished")
#
# print("Test model")
# Y_pred_test = model.predict(x_test)
# Y_pred_train = model.predict(x_train)
#
# acc = accuracy_score(y_test, Y_pred_test)
# acc2 = accuracy_score(y_train, Y_pred_train)
# print("Accuracy on test:", acc)
# print(classification_report(y_test, Y_pred_test, target_names=model.classes_))
# print("Accuracy on train:", acc2)
# print(classification_report(y_train, Y_pred_train, target_names=model.classes_))
# a1 = []
# aval = []
# atest = []

# for i in ['identity', 'logistic', 'tanh', 'relu']:
#     model = MLPClassifier(max_iter=10000, hidden_layer_sizes=(200,), solver='sgd', activation=i)
#     model.fit(x_train, y_train)
#     print("Train finished")
#     print("Test model")
#     Y_pred_val = model.predict(x_test2)
#     Y_pred_test = model.predict(x_test)
#     Y_pred_train = model.predict(x_train)
#     acc_val = accuracy_score(y_test2, Y_pred_val)
#     acc_test = accuracy_score(y_test, Y_pred_test)
#     acc = accuracy_score(y_train, Y_pred_train)
#     a1.append(float(acc))
#     aval.append(float(acc_val))
#     atest.append(float(acc_test))
#     print("activation: ", i, "--", "Accuracy on train :", acc)
#     print("activation: ", i, "--", "Accuracy on test :", acc_test)
#     print("activation: ", i, "--", "Accuracy on test :", acc_val)
#
# plt.plot(['identity', 'logistic', 'tanh', 'relu'], a1, "ko-", label='train')
# plt.plot(['identity', 'logistic', 'tanh', 'relu'], aval, "g*-", label='val')
# plt.plot(['identity', 'logistic', 'tanh', 'relu'], atest, "red", label='test')
# plt.xlabel('activation', fontsize=12)
# plt.ylabel('mean score', fontsize=12)
# plt.title('Hyperparameter Tuning', pad=15, size=15)
# plt.title('Hyperparameter tuning of activation', pad=15, size=15)
# plt.legend(loc='best', labelspacing=2, handlelength=3, fontsize=10, shadow=True)
# plt.show()

# print("Start mlp")
# model = SVC(max_iter=100000)
# # model = MLPClassifier(max_iter=100000)
# # model = RandomForestClassifier()
#
# model.fit(x_train, y_train)
# print("Train finished")
#
# print("Test model")
# Y_pred_test = model.predict(x_test)
# # Y_pred_train = model.predict(x_train)
#
# acc = accuracy_score(y_test, Y_pred_test)
# end_time = time.time()
# total_time = end_time - start_time
# print("using time：", total_time, "s")
# # acc2 = accuracy_score(y_train, Y_pred_train)
# # model.save("MLP")
#
# print("Accuracy on test:", acc)
# print(classification_report(y_test, Y_pred_test, target_names=model.classes_))



# print("Start train SVC")
# model = SVC(max_iter=100000)
# model.fit(x_train, y_train)
# print("Train finished")
#
# # 测试
# print("Test model")
# Y_pred_test = model.predict(x_test)
# # Y_pred_train = model.predict(X_train)
#
# acc = accuracy_score(y_test, Y_pred_test)
# # # acc2 = accuracy_score(Y_train, Y_pred_train)
# #
# end_time = time.time()
# total_time = end_time - start_time
# print("using time：", total_time, "s")
# print("Accuracy on test:", acc)
# print(classification_report(y_test, Y_pred_test, target_names=model.classes_))


# print("Accuracy on train:", acc2)
# print(classification_report(y_train, Y_pred_train, target_names=model.classes_))
# print("Accuracy on train:", acc2)
# print(classification_report(y_train, Y_pred_train, target_names=model.classes_))
#
# print("start grid search: test-mlp")
# parameters = {
#     'hidden_layer_sizes': [(50,), (75,), (125,), (175,), (200,), (100, 50), (100, 100)],
#     'activation': ['identity', 'logistic', 'tanh', 'relu'],
#     'solver': ['lbfgs', 'sgd', 'adam']}
#
# svm = MLPClassifier(max_iter=10000)
# svm_m = GridSearchCV(svm, parameters, verbose=2, return_train_score=True)
# svm_m.fit(x_train, y_train)
# sorted(svm_m.cv_results_.keys())
# print(svm_m.best_params_)
# plt.plot(['lbfgs', 'sgd’', 'adam'], svm_m.cv_results_["mean_train_score"], "ko-", label='train')
# plt.plot(['lbfgs', 'sgd’', 'adam'], svm_m.cv_results_["mean_test_score"], "g*-", label='test')
# plt.xlabel('solver', fontsize=12)
# plt.ylabel('mean score', fontsize=12)
# plt.title('Hyperparameter Tuning', pad=15, size=15)
# plt.title('Hyperparameter tuning of solver', pad=15, size=15)
# plt.legend(loc='best', labelspacing=2, handlelength=3, fontsize=10, shadow=True)
# plt.show()


# x_train2 = get_feature(train, w2v_model)
# x_test2 = get_feature(val, w2v_model)
# y_train2 = train['author']
# y_test2 = val['author']
#
# print("start grid search: val-mlp")
# parameters = {
#     'hidden_layer_sizes': [(50,), (75,), (125,), (175,), (200,), (100, 50), (100, 100)],
#     'activation': ['identity', 'logistic', 'tanh', 'relu'],
#     'solver': ['lbfgs', 'sgd', 'adam']}
# svm = MLPClassifier(max_iter=10000)
# svm_m = GridSearchCV(svm, parameters, verbose=2, return_train_score=True)
# svm_m.fit(x_train2, y_train2)
# sorted(svm_m.cv_results_.keys())
# print(svm_m.best_params_)
# plt.plot(['lbfgs', 'sgd’', 'adam'], svm_m.cv_results_["mean_train_score"], "ko-", label='train')
# plt.plot(['lbfgs', 'sgd’', 'adam'], svm_m.cv_results_["mean_test_score"], "g*-", label='test')
# plt.xlabel('solver', fontsize=12)
# plt.ylabel('mean score', fontsize=12)
# plt.title('Hyperparameter Tuning', pad=15, size=15)
# plt.title('Hyperparameter tuning of solver', pad=15, size=15)
# plt.legend(loc='best', labelspacing=2, handlelength=3, fontsize=10, shadow=True)
# plt.show()

# print("start grid search: SVM")
# parameters = {
#     'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
#     'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.7, 1, 1.2, 1.5, 1.8, 2, 2.4, 2.8, 3, 3.5, 3.8, 4],
#     'tol': [1e-3, 1e-4, 1e-5, 1e-2, 1e-1],
#     'decision_function_shape': ['ovo', 'ovr']}
# svm = SVC(max_iter=100000)
# svm_m = GridSearchCV(svm, parameters, verbose=2, return_train_score=True)
# svm_m.fit(x_train, y_train)
# sorted(svm_m.cv_results_.keys())
# print(svm_m.best_params_)
