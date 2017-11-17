import csv
import sys
import os
import numpy as np
import keras

def read_test_data(test_data_path):
    x = []
    with open(test_data_path, 'r') as test_data:
        test_data.readline()
        for line in test_data:
            label, pixels = line.strip().split(',')
            pixels = list(map(int, pixels.split(' ')))
            x.append(pixels)
    return x

def write_answer(answer_path, pred):
    if os.path.exists(answer_path):
        os.remove(answer_path)
    with open(answer_path, 'a') as ans:
        ans.write('id,label\r\n')
        id = 0
        for vec in pred:
            str = '%d,%d\r\n'% (id, list(vec).index(max(vec)))
            id += 1
            ans.write(str)

def main():
    img_rows, img_cols = 48, 48
    test_data_path = sys.argv[1]
    answer_path = sys.argv[2]
    model_path = './two.h5?dl=1'
    model = keras.models.load_model(model_path)
    x_test = read_test_data(test_data_path)
    x_test = np.array(x_test).reshape(np.array(x_test).shape[0], img_rows, img_cols, 1)
    x_test = x_test.astype('float32')
    x_test /= 255
    pred = model.predict(x_test)
    write_answer(answer_path, pred)

if __name__ == '__main__':
	main()