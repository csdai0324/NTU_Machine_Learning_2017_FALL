import os
import sys
import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing import sequence

def indexing_testing_data(testing_data_path, testing_data_index_path, dict_path):
    d = read_dict(dict_path)
    print('Indexing testing data ... ')
    with open(testing_data_path, 'r') as test_data:
        all_index_text = []
        test_data.readline()
        counter = 0
        for line in test_data:
            counter += 1
            first_comma = line.index(',')
            index = line.strip()[:first_comma]
            text = line.strip()[first_comma+1:]
            index_text = ''
            words = text.split(' ')
            for word in words:
                if word in d:
                    index_text += str(d.index(word) + 1) +' '
                else:
                    index_text += str(1) +' ' 
            all_index_text.append(index_text.strip())
        if os.path.isfile(testing_data_index_path):
            os.remove(testing_data_index_path)
        with open(testing_data_index_path, 'a') as index_data:
            index = 0
            for index_text in all_index_text:
                index_data.write(str(index) +','+ index_text +'\r\n')
                index += 1
    return None

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

def read_dict(dict_path):
    d = []
    with open(dict_path, 'r') as dict_txt:
        for line in dict_txt:
            d.append(line.strip())
    return d

def main():
    testing_data_path = sys.argv[1]
    answer_path = sys.argv[2]
    testing_data_index_path = './testing_data_index.txt'
    model_path = './model_7.h5?dl=1'
    dict_path = './dict.txt'
    max_review_length = 30
    indexing_testing_data(testing_data_path, testing_data_index_path, dict_path)
    x_test = read_test_data(testing_data_index_path)
    x_test = sequence.pad_sequences(x_test, maxlen=max_review_length, padding='post')
    model = load_model(model_path)
    pred = model.predict(x_test)
    write_answer(pred, answer_path)

if __name__ == '__main__':
    main()


