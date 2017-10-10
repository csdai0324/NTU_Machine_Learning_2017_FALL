import csv
import os
import sys
import datetime
import math
import numpy as np

feature_num = 18
feature_list = ['AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2', 'NOx',
                'O3','PM10', 'PM2.5', 'RAINFALL', 'RH', 'SO2', 'THC',
                'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR']
train_data_path = './train.csv'
test_data_path = './test.csv'
output_answer_path = './answer.csv'

class Data(object):
    
    def __init__(self, location, date, feature_dict):
        
        self.location = location
        self.date = date
        self.feature_dict = feature_dict
        self.str2num()
        self.PM2point5 = self.feature_dict['PM2.5']
        
    def str2num(self):   
        for feature in feature_list:
            if feature == 'RAINFALL':
                temp = self.feature_dict[feature]
                for i in range(len(temp)):
                    if self.feature_dict[feature][i] == 'NR':
                        self.feature_dict[feature][i] = 0
                    else:
                        self.feature_dict[feature][i] = 0
                self.feature_dict[feature] = np.array(self.feature_dict[feature])
                continue
            temp = self.feature_dict[feature]
            for i in range(len(temp)):
                self.feature_dict[feature][i] = float(temp[i])
            self.feature_dict[feature] = np.array(self.feature_dict[feature])
            
    def feature2matrix(self):   
        x = np.ndarray(shape=(18, 24), dtype=float)
        cntr = 0
        for feature in feature_list:
            x[cntr] = self.feature_dict[feature]
            cntr += 1
            
        return np.asmatrix(x)
    
    def test_feature2matrix(self):      
        x = np.ndarray(shape=(18, 9), dtype=float)
        cntr = 0
        for feature in feature_list:
            x[cntr] = self.feature_dict[feature]
            cntr += 1
            
        return np.asmatrix(x)
    
    def get_matrix(self):
        self.x = self.feature2matrix()
        return self.x
    
    def test_get_matrix(self):
        self.x = self.test_feature2matrix()
        return self.x
    
    def get_PM2point5(self):
        return self.PM2point5
    
def PM2point5_dataset(train_data_path, fs):
    train_data, feature_mean_dict, feature_std_dict = feature_scaling(read_train_data(train_data_path), fs)
    x_matrix = None
    y_matrix = None
    cntr = 0
    for data in train_data:
        if cntr == 0:
            x_matrix = data.get_matrix()
            y_matrix = data.get_PM2point5()
            cntr += 1
            continue
        x_temp = data.get_matrix()
        y_temp = data.get_PM2point5()
        x_matrix = np.hstack((x_matrix, x_temp))
        y_matrix = np.hstack((y_matrix, y_temp))
        cntr += 1
    
    return x_matrix, y_matrix, feature_mean_dict, feature_std_dict

def read_train_data(train_data_path):
    train_data = []
    feature_value = []
    line_count = 0
    
    with open(train_data_path, encoding='big5') as train_data_csv:
        next(train_data_csv)
        reader = csv.reader(train_data_csv, delimiter=',')
        for line in reader:
            line_count += 1
            time = line[0].split('/')
            date = datetime.datetime(int(time[0]), int(time[1]), int(time[2]))
            location = line[1].strip()
            value = []
            for ele in line[3:]:
                value.append(ele)     
            feature_value.append(value)
            if line_count % 18 == 0:
                feature_dict = dict(zip(feature_list, feature_value))
                train_data.append(Data(location, date, feature_dict))
                feature_value = []
                
    return train_data

def read_test_data(test_data_path, feature_mean_dict, feature_std_dict, fs):
    test_data = []
    feature_value = []
    line_count = 0
    with open(test_data_path, encoding='big5') as test_data_csv:
        reader = csv.reader(test_data_csv, delimiter=',')
        for line in reader:
            line_count += 1
            value = []
            for ele in line[2:]:
                value.append(ele)     
            feature_value.append(value)
            if line_count % 18 == 0:
                feature_dict = dict(zip(feature_list, feature_value))
                test_data.append(Data('', '0-0-0', feature_dict))
                feature_value = [] 
    
    if fs == 1:
        for data in test_data:
            for feature in feature_list:
                if feature == 'RAINFALL':
                    data.feature_dict[feature] = 0
                    continue
                data.feature_dict[feature] = (data.feature_dict[feature] - feature_mean_dict[feature]) / feature_std_dict[feature]
    else:
        for data in test_data:
            for feature in feature_list:
                if feature == 'RAINFALL':
                    data.feature_dict[feature] = 0

    return test_data

def feature_scaling(train_data, fs):  
    feature_value = [np.array([])]*24
    feature_dict = dict(zip(feature_list, feature_value))
    feature_mean = []*24
    feature_mean_dict = dict(zip(feature_list, feature_value))
    feature_std = []*24
    feature_std_dict = dict(zip(feature_list, feature_value)) 
    
    for data in train_data:    
        for feature in feature_list:
            feature_dict[feature] = np.concatenate((feature_dict[feature], data.feature_dict[feature]))   
    for feature in feature_list:    
        if feature == 'RAINFALL':
            feature_mean_dict[feature] = 0
            feature_std_dict[feature] = 1
            continue
        if fs == 1:
            feature_mean_dict[feature] = feature_dict[feature].mean()
            feature_std_dict[feature] = feature_dict[feature].std()
        else:
            feature_mean_dict[feature] = 0
            feature_std_dict[feature] = 1
    
    feature_dict = dict(zip(feature_list, feature_value))
    for data in train_data:
        for feature in feature_list:
            if feature == 'RAINFALL':
                data.feature_dict[feature] = 0
                continue
            data.feature_dict[feature] = (data.feature_dict[feature] - feature_mean_dict[feature]) / feature_std_dict[feature]

    return train_data, feature_mean_dict, feature_std_dict


def test(test_data, W_b):
    answer = []
    for data in test_data:
        x_matrix = data.test_get_matrix()
        #a = np.array(x_matrix[9]).reshape(-1)
        #b = np.array(x_matrix[7]).reshape(-1)
        #c = np.concatenate((a, b), axis=0)
        c = np.array(x_matrix[7:10]).reshape(-1)
        x_ = np.concatenate(([1], c), axis=0)
        a = np.dot(W_b, x_) + 2.5
        if a % 1 > 0.5:
            a = int(a) + 1
        else:
            a = int(a)
        answer.append(int(a))
    return answer

def output_answer(output_answer_path, answer):
    #os.remove(output_answer_path)
    #os.mknod("answer.csv")
    with open(output_answer_path, 'a') as output_csv:
        output_csv.write('id,value\r\n')
        for i in range(len(answer)):
            line = 'id_%d,%f\r\n' % (i, answer[i])
            output_csv.write(line)

def gradient_descent(X, Y, regularization_lambda, learning_rate):
    x_train = []
    y_train = []
    hour = 9
    epoch = 100000
    feature_use = 1
    for i in range(X.shape[1] - hour - 1):
        if (i + hour) % 480 >= (480 - hour):
            continue
        a = np.array(X[9,i:i+hour]).reshape(-1)
        #b = np.array(X[7,i:i+hour]).reshape(-1)
        #c = np.array(X[7:10,i:i+hour]).reshape(-1)
        x_train.append(np.concatenate(([1], a), axis=0))
        y_train.append(Y[i+hour])
    
    print(x_train[0])
    W = np.random.uniform(-1, 1, (feature_use*hour+1))
    #W = [-0.049537543635841841, -0.017419422039271885, 0.016669280080405299, -0.0046551502571262823, -0.0084486241595227128, 0.0075641775368446252, -0.024523643810010681, 0.0040900367166305891, -0.019807559442639348, 0.067423967149425459, 0.0039491222374465123, 0.0075671650413279834, -0.025599351385903121, 0.032201475525498888, -0.01408729145824795, -0.019025877502372557, 0.020613216120095649, -0.015708789905703263, 0.07498745101957538, -0.036284239517546737, -0.0049864938061120537, 0.20986194133910363, -0.23914105686084783, -0.027507316144628469, 0.50050330428874967, -0.57564933375077254, 0.018905893308170458, 0.99348142840239184]

    temp_W = np.random.rand(feature_use*hour+1)
    example_num = float(len(x_train))
    
    for e in range(epoch):
        loss = []
        RMSD = 0.
        temp_W = W
        for i in range(int(example_num) - 1):
            loss.append((np.dot(W, x_train[i]) - y_train[i])**2)
        
        for j in range(feature_use*hour + 1):
            delta = 0
            if j == 0:
                for i in range(int(example_num) - 1):
                    delta += (y_train[i] - np.dot(W, x_train[i])) * (-1)
                temp_W[j] = W[j] - learning_rate * (1. / example_num) * delta
            else:
                for i in range(int(example_num) - 1):
                    delta += (y_train[i] - np.dot(W, x_train[i])) * (-1) * x_train[i][j]
                temp_W[j] = W[j]*(1-regularization_lambda*(1. / example_num)*W[j]) - learning_rate * (1. / example_num) * delta                                 

        W = temp_W
        if (e) % 10 == 0:
            print('epoch', e+1,'loss:', sum(loss), 'RMSD:', math.sqrt(sum(loss)/example_num))
            if (e) % 500 == 0:
                print(list(W))
                fname = './W/W_'+ str(e)
                with open(fname, 'w') as f:
                	f.write(str(list(W)))
        
    return W

if __name__ == '__main__':

    train_data_path = './train.csv'
    output_answer_path = sys.argv[2]
    test_data_path = sys.argv[1]
    X, Y, feature_mean_dict, feature_std_dict = PM2point5_dataset(train_data_path, 0)
    X_ = read_test_data(test_data_path,feature_mean_dict, feature_std_dict, 0)
    W_X = [-0.049537543635841841, -0.017419422039271885, 0.016669280080405299, -0.0046551502571262823, -0.0084486241595227128, 0.0075641775368446252, -0.024523643810010681, 0.0040900367166305891, -0.019807559442639348, 0.067423967149425459, 0.0039491222374465123, 0.0075671650413279834, -0.025599351385903121, 0.032201475525498888, -0.01408729145824795, -0.019025877502372557, 0.020613216120095649, -0.015708789905703263, 0.07498745101957538, -0.036284239517546737, -0.0049864938061120537, 0.20986194133910363, -0.23914105686084783, -0.027507316144628469, 0.50050330428874967, -0.57564933375077254, 0.018905893308170458, 0.99348142840239184]
    answer = test(X_, W_X)
    output_answer(output_answer_path, answer)