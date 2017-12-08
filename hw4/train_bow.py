import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.normalization import BatchNormalization

training_label_data_path = '/home/csdai/hw4_data/training_label.txt'
training_label_index_data_path = '/home/csdai/hw4_data/training_label_index.txt'
nolabel_data_path = '/home/csdai/hw4_data/training_nolabel.txt'
nolabel_index_data_path = '/home/csdai/hw4_data/training_nolabel_index.txt'
testing_data_path = '/home/csdai/hw4_data/testing_data.txt'
labeled_index_data_path = '/home/csdai/hw4_data/labeled_index.txt'
dict_path = '/home/csdai/hw4_data/dict.txt'
testing_data_index_path = '/home/csdai/hw4_data/testing_data_index.txt'
answer_path = '/home/csdai/hw4_data/answer.txt'
bow_answer_path = '/home/csdai/hw4_data/bow_answer.txt'
model_path = '/home/csdai/hw4_data/model/'

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
    #print(count_1, count_0)
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
            
def read_dict(dict_path):
    d = []
    with open(dict_path, 'r') as dict_txt:
        for line in dict_txt:
            d.append(line.strip())
    return d

def model_BOW_1(input_size):
    model = Sequential()
    model.add(Dense(1024, activation='linear', input_shape=(input_size, )))
    model.add(Dropout(0.3))
    model.add(LeakyReLU(alpha=0.3))
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='linear'))
    model.add(Dropout(0.3))
    model.add(LeakyReLU(alpha=0.3))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
              metrics=['accuracy'])
    print(model.summary())
    return model

def main():
    max_review_length = 30
    val_cut = 10000
    x_train, y_train = read_train_data(training_label_index_data_path)
    d = read_dict(dict_path)

    x_train_vector = []
    for x in x_train:
        x_vector = np.zeros(len(d) + 1)
        for ele in x:
            x_vector[ele] = 1
        x_train_vector.append(x_vector)

    model = model_BOW_1(len(d) + 1)
    epoch = 100
    batch_size = 128
    for e in range(epoch):
        for i in range(int(len(x_train)/batch_size)):
            if (i+1)*batch_size > 19999:
                continue
            if (i + 1) % 20 == 0:
                print('epoch:', (e+1), ' batch:', (i+1))
            x_train_vector_batch = x_train_vector[i*batch_size:(i+1)*batch_size]
            y_train_batch = y_train[i*batch_size:(i+1)*batch_size]
            x_train_vector_batch = np.array(x_train_vector_batch).reshape(np.array(x_train_vector_batch).shape[0], len(d) + 1)
            model.train_on_batch(x_train_vector_batch, y_train_batch)
    model.save(model_path +'bow_model.h5')

if __name__ == '__main__':
    main()