import json
import re
import tensorflow as tf
import tqdm
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

# Define hyperparameters
max_features = 2000  # Maximum number of words to keep, based on word frequency
max_len = 100  # Maximum length of input sequence
embedding_dim = 128  # Dimension of embedding vector
lstm_units = 64  # Number of LSTM units
num_classes = 10  # Number of output classes

# 从JSON文件中读取文本和标签
texts_train = []
labels_train = []
texts_test = []
labels_test = []
texts_val = []
labels_val = []

print("Loading data")
with open('darkreddit_authorship_attribution_anon/darkreddit_authorship_attribution_train_anon.jsonl', 'r') as f:
    for line in f:
        line = json.loads(line)
        for key, value in line.items():
            if key == 'author':
                labels_train.append(value)
            elif key == 'comment':
                texts_train.append(value)
    f.close()

with open('darkreddit_authorship_attribution_anon/darkreddit_authorship_attribution_test_anon.jsonl', 'r') as f:
    for line in f:
        line = json.loads(line)
        for key, value in line.items():
            if key == 'author':
                labels_test.append(value)
            elif key == 'comment':
                texts_test.append(value)
    f.close()

with open('darkreddit_authorship_attribution_anon/darkreddit_authorship_attribution_val_anon.jsonl', 'r') as f:
    for line in f:
        line = json.loads(line)
        for key, value in line.items():
            if key == 'author':
                labels_val.append(value)
            elif key == 'comment':
                texts_val.append(value)
    f.close()
print("Loading finished")
# print(len(texts_train))
# print(len(labels_train))
# print(len(texts_test))
# print(len(labels_test))
# print(len(texts_val))
# print(len(labels_val))

X_train_pre = []
X_test_pre = []
X_val_pre = []

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

for text in tqdm.tqdm(texts_val, desc="Processing val data "):
    text = re.sub(r'http\S+', 'URL', text)
    # text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[@$%^&*()]', '', text)
    text = re.sub(r'\s+', ' ', text)
    X_val_pre.append(text)

# Convert sentance to int sequences
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_features)
all_text = X_test_pre + X_val_pre + X_train_pre
tokenizer.fit_on_texts(all_text)
X_train_pre = tokenizer.texts_to_sequences(X_train_pre)
X_test_pre = tokenizer.texts_to_sequences(X_test_pre)
X_val_pre = tokenizer.texts_to_sequences(X_val_pre)

# Convert lable to numerical labels
label_map = {'user1': 0, 'user2': 1, 'user3': 2, 'user4': 3, 'user5': 4, 'user6': 5, 'user7': 6, 'user8': 7, 'user9': 8,
             'user10': 9}
Y_train_num = [label_map[label] for label in labels_train]
Y_test_num = [label_map[label] for label in labels_test]
Y_val_num = [label_map[label] for label in labels_val]

# Preprocess input sequences (i.e., pad/truncate sequences to max_len)
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train_pre, maxlen=max_len)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test_pre, maxlen=max_len)
X_val = tf.keras.preprocessing.sequence.pad_sequences(X_val_pre, maxlen=max_len)

# Convert labels to one-hot vectors
Y_train = tf.keras.utils.to_categorical(Y_train_num, num_classes)
Y_test = tf.keras.utils.to_categorical(Y_test_num, num_classes)
Y_val = tf.keras.utils.to_categorical(Y_val_num, num_classes)

# print(X_train.shape)
# print(X_val.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test.shape)
# print(Y_val.shape)

# Build model
model = Sequential()
model.add(Embedding(max_features, embedding_dim, input_length=max_len))
model.add(LSTM(units=lstm_units, return_sequences=True, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(LSTM(units=lstm_units, activation='sigmoid'))  #
model.add(Dropout(0.2))
model.add(Dense(units=num_classes, activation='softmax'))  #  Because the table has been converted to a 10-dimensional vector, the output dimension of the last layer is the number of categories

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=64)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, Y_test, batch_size=64)
print('Test accuracy:', test_acc)


