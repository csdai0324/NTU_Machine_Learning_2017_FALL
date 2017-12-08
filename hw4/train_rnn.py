import os
import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense, LSTM, LeakyReLU, Dropout, Conv1D, MaxPooling1D, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.normalization import BatchNormalization


training_label_index_data_path = './training_label_index.txt'
nolabel_index_data_path = './training_nolabel_index.txt'
labeled_index_data_path = './labeled_index.txt'
dict_path = './dict.txt'

def read_train_data(training_label_index_data_path):
    x = []
    y = []
    with open(training_label_index_data_path, 'r') as train_data:
        for line in train_data:
            label, text = line.strip().split(',')
            x.append(list(map(int, text.split(' '))))
            y.append(int(label))
    return x, y

def read_test_data(testing_data_index_path):
    x = []
    with open(testing_data_index_path, 'r') as test_data:
        for line in test_data:
            index, text = line.strip().split(',')
            x.append(list(map(int, text.split(' '))))
    return x

def write_answer(pred, answer_path):
    if os.path.isfile(answer_path):
        os.remove(answer_path)
    with open(answer_path, 'a') as answer:
        index = 0
        answer.write('id,label\r\n')
        for score in pred:
            if score >= 0.5:
                answer.write(str(index) +',1\r\n')
            else:
                answer.write(str(index) +',0\r\n')
            index += 1 

def read_nolabel_data(nolabel_index_data_path):
    x = []
    with open(nolabel_index_data_path, 'r') as train_data:
        for line in train_data:
            x.append(list(map(int, line.strip().split(' '))))
    return x

def read_labeled_data(labeled_index_data_path, threshold):
    x = []
    y = []
    count_1 = 0
    count_0 = 0
    with open(labeled_index_data_path, 'r') as labeled_data:
        for line in labeled_data:
            label, text = line.strip().split(',')
            if float(label) >= threshold:
                label = 1
                x.append(list(map(int, text.split(' '))))
                y.append(int(label))
                count_1 += 1
            elif float(label) <= 1 - threshold:
                label = 0
                x.append(list(map(int, text.split(' '))))
                y.append(int(label))
                count_0 += 1
    print(count_1, count_0)
    return x, y

def generate_label(pred, x, labeled_index_data_path):
    if os.path.isfile(labeled_index_data_path):
        os.remove(labeled_index_data_path)
    with open(labeled_index_data_path, 'a') as labeled_data:
        for cntr in range(len(pred)):
            s = ''
            for ele in x[cntr]:
                s += str(ele) +' '
            labeled_data.write(str(pred[cntr][0]) +','+ s.strip() +'\r\n')

def model_1():
    model = Sequential()
    embedding_vecor_length = 256
    max_word = 15000
    model.add(Embedding(max_word, embedding_vecor_length, input_length=max_review_length, mask_zero=True))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    print(model.summary())
    return model

def model_2():
    model = Sequential()
    embedding_vecor_length = 512
    max_word = 15000
    model.add(Embedding(max_word, embedding_vecor_length, input_length=max_review_length, mask_zero=True))
    model.add(Dropout(0.2))
    model.add(LSTM(512, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def model_3():
    model = Sequential()
    embedding_vecor_length = 100
    max_word = 15000
    model.add(Embedding(max_word, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def model_4():
    model = Sequential()
    embedding_vecor_length = 256
    max_word = 15000
    model.add(Embedding(max_word, embedding_vecor_length, input_length=max_review_length))
    model.add(Dropout(0.25))
    model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def main():

	max_review_length = 30
	val_cut = 50000
	x_train, y_train = read_train_data(training_label_index_data_path)
	x_labeled, y_labeled = read_labeled_data(labeled_index_data_path, 0.95)
	x_train = x_train + x_labeled
	y_train = y_train + y_labeled
	x_val, y_val = x_train[:val_cut], y_train[:val_cut]
	x_val = sequence.pad_sequences(x_val, maxlen=max_review_length, padding='post')
	x_train = sequence.pad_sequences(x_train, maxlen=max_review_length, padding='post')
	model = model_3()
	epoch = 1
	model.fit(x_train, y_train, epochs=epoch, batch_size=2048, validation_data=(x_val, y_val))

if __name__ == '__main__':
    main()