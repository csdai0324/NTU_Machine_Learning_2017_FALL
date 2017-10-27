import sys
import os

import numpy as np

def read_train_data(train_data_path):
    x_1 = []
    x_2 = []
    y = []
    with open(train_data_path[0], 'r') as X_train, \
         open(train_data_path[1], 'r') as Y_train:
            X_train.readline()
            Y_train.readline()
            for x_, y_ in zip(X_train, Y_train):
                if int(y_) == 1:
                    x_1.append(list(map(int, x_.strip().split(','))))
                else:
                    x_2.append(list(map(int, x_.strip().split(','))))
                y.append(int(y_))
    return x_1, x_2, y

def read_test_data(test_data_path):
    x_test = []
    with open(test_data_path, 'r') as X_test:
        X_test.readline()
        for x_ in X_test:
            x_test.append(list(map(int, x_.strip().split(','))))
    return x_test

def feature_normalization(x_1, x_2):
    x = x_1 + x_2
    feature_mean = np.mean(x, axis=0)
    feature_std = np.std(x, axis=0)
    for i in range(len(x_1)):
        x_1[i] = (x_1[i] - feature_mean) / feature_std
    for i in range(len(x_2)):
        x_2[i] = (x_2[i] - feature_mean) / feature_std
    return x_1, x_2, feature_mean, feature_std

def test_feature_normalization(x, feature_mean, feature_std):
    for i in range(len(x)):
        x[i] = (x[i] - feature_mean) / feature_std
    return x

def sigmoid(z):
    res = 1 / (1. + np.exp(-z))
    return np.clip(res, 0.000000001, 0.999999999)

def predict(X_test, mu1, mu2, shared_sigma, N1, N2):
    sigma_inverse = np.linalg.inv(shared_sigma)
    w = np.dot((mu1 - mu2), sigma_inverse)
    x = np.matrix(X_test).T
    b = (-0.5 * np.dot(np.dot([mu1], sigma_inverse), mu1)) + (-.5 * np.dot(np.dot([mu2], sigma_inverse), mu2)) + np.log(float(N1)/N2)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    return y

def write_answer(answer):
    if os.path.exists(answer_path):
        os.remove(answer_path)
    with open(answer_path, 'a') as ans:
        ans.write('id,label\r\n')
        id = 1
        for y in answer:
            if y >= 0.5:
                str ='%d,%d\r\n' % (id, 1)
                ans.write(str)
            elif y < 0.5:
                str ='%d,%d\r\n' % (id, 0)
                ans.write(str)
            id += 1

if __name__ == '__main__':
	x_train = sys.argv[3]
	y_train = sys.argv[4]
	train_data_path = [x_train, y_train]
	test_data_path = sys.argv[5]
	answer_path = sys.argv[6]
	x1, x2, y = read_train_data(train_data_path)
	x1, x2, feature_mean, feature_std = feature_normalization(x1, x2)
	x = x1 + x2
	mu1 = np.mean(x1, axis=0)
	mu2 = np.mean(x2, axis=0)
	sigma1 = np.zeros((106, 106))
	sigma2 = np.zeros((106, 106))
	for i in range(len(x)):
	    if y[i] == 1:
	        sigma1 += np.dot(np.transpose([x[i] - mu1]), [(x[i] - mu1)])
	    else:
	        sigma2 += np.dot(np.transpose([x[i] - mu2]), [(x[i] - mu2)])
	sigma1 /= len(x1)
	sigma2 /= len(x2)
	shared_sigma = (float(len(x1)) / len(x)) * sigma1 \
	                + (float(len(x2))/ len(x)) * sigma2
	x_test = read_test_data(test_data_path)
	x_test = test_feature_normalization(x_test, feature_mean, feature_std)
	ans = predict(x_test, mu1, mu2, shared_sigma, len(x1), len(x2))
	write_answer(ans.A1)