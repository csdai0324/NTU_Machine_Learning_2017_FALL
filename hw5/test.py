import json
import sys
import os
import numpy as np 
from keras.models import load_model
from keras.models import model_from_json

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

def load_model(model_json_path, model_weight_path):
	with open(model_json_path) as json_file:
		model_json = json.load(json_file)
		model_ = model_from_json(model_json)
		model_.load_weights(model_weight_path)
	return model_

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

def main():
	model_json_path = './model/model_json.txt'
	model_weigth_path = './model/model.h5'
	test_data_path = sys.argv[1]
	answer_path = sys.argv[2]
	test_users, test_movies = read_test_data(test_data_path)
	model1 = load_model(model_json_path, model_weigth_path)
	pred = predict(answer_path, model1, test_data_path)

if __name__ == '__main__':
	main()
