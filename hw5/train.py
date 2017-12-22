import os
import json
import numpy as np
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense
from keras.models import Sequential, load_model
from keras.models import model_from_json
from keras.regularizers import l2

train_data_path = './hw5_data/train.csv'
test_data_path = '/.hw5_data/test.csv'

def read_train_data(train_data_path):
    with open(train_data_path, 'r') as train_data:
        train_data.readline()
        users = []
        movies = []
        ratings = []
        for data in train_data:
            lt = data.strip().split(',')
            users.append(int(lt[1]))
            movies.append(int(lt[2]))
            ratings.append(int(lt[3]))
    return users, movies, ratings

def read_test_data(test_data_path):
    with open(test_data_path, 'r') as test_data:
        test_data.readline()
        users = []
        movies = []
        for data in test_data:
            lt = data.strip().split(',')
            users.append(int(lt[1]))
            movies.append(int(lt[2]))
    return users, movies

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def standard_normalization(ratings):
    mean = np.array(ratings).mean()
    std = np.array(ratings).std()
    nor_ratings = []
    for rating in ratings:
        nor_ratings.append((rating - mean) / std)
    return nor_ratings, mean, std

def save_model(model):
    json_string = model.to_json()
    with open('/home/csdai0324/hw5_data/model_json.txt', 'w') as outfile:  
        json.dump(json_string, outfile)
    model.save_weights('/home/csdai0324/hw5_data/model.h5')

def load_model(model_json_path, model_weight_path):
    with open(model_json_path) as json_file:  
        model_json = json.load(json_file)
    model_ = model_from_json(model_json)
    model_.load_weights(model_weight_path)
    return model_

def predict_rating(userid, movieid, model):
    return model.predict([np.array([userid]), np.array([movieid])])[0][0]

def predict(answer_path, model, test_data_path):
    if os.path.isfile(answer_path):
        os.remove(answer_path)
    test_users, test_movies = read_test_data(test_data_path)
    pred = []
    for i in range(len(test_users)):
        pred.append(predict_rating(test_users[i], test_movies[i], model))
    with open(answer_path, 'a') as answer:
        answer.write('TestDataID,Rating\r\n')
        index = 1
        for ele in pred:
            answer.write(str(index) +','+ str(ele) +'\r\n')
            index += 1
    return pred

def nor_predict(answer_path, model, test_data_path, mean, std):
    if os.path.isfile(answer_path):
        os.remove(answer_path)
    test_users, test_movies = read_test_data(test_data_path)
    pred = []
    for i in range(len(test_users)):
        pred.append((predict_rating(test_users[i], test_movies[i], model) * std) + mean)
    with open(answer_path, 'a') as answer:
        answer.write('TestDataID,Rating\r\n')
        index = 1
        for ele in pred:
            answer.write(str(index) +','+ str(ele) +'\r\n')
            index += 1
    return pred

def mf_model(user_num, movie_num, factors):
    u = Sequential()
    u.add(Embedding(user_num + 1, factors, input_length=1, W_regularizer=l2(1e-6)))
    u.add(Reshape((factors,)))
    m = Sequential()
    m.add(Embedding(movie_num + 1, factors, input_length=1, W_regularizer=l2(1e-6)))
    m.add(Reshape((factors,)))
    model = Sequential()
    model.add(Merge([u, m], mode='dot'))
    model.compile(loss='mse', optimizer='adamax')
    return model

def mf_dnn_model(user_num, movie_num, factors):
    u = Sequential()
    u.add(Embedding(user_num + 1, factors, input_length=1, W_regularizer=l2(1e-6)))
    u.add(Reshape((factors,)))
    m = Sequential()
    m.add(Embedding(movie_num + 1, factors, input_length=1, W_regularizer=l2(1e-6)))
    m.add(Reshape((factors,)))
    model = Sequential()
    model.add(Merge([u, m], mode='concat'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adamax')
    return model

def main():
	users, movies, ratings = read_train_data(train_data_path)
	ratings, mean, std = standard_normalization(ratings)
	model = mf_dnn_model(max(users), max(movies), 256)
	model.fit([np.array(users), np.array(movies)], np.array(ratings), batch_size=512, nb_epoch=1, verbose=1)
	save_model(model)

if __name__ == '__main__':
	main()

