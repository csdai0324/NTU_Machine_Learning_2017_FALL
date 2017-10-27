import sys
import os

import numpy as np

theta = [-2.2279712282868114, 1.3657255409957787, 0.12326313513236879, 0.32918023679192282, 2.4765636275493965, 0.25958502418295609, 0.99547192902140291, -0.28177628552566575, -0.57430952702509552, -0.037015473030534478, -0.9807268958165144, -0.34916715105984081, -0.68535132841848967, -0.47900660681087276, -0.13359658954284473, -0.30118526978007365, -0.241761670860351, -0.25503588460783805, -0.11007231101962915, -0.11847571337570444, -0.15702528761943457, -0.25953216785105687, -0.20124172258766848, -0.028515278765544164, -0.029444490958160072, 0.16738668556411712, 0.17494130324121843, -0.31433452926515038, 0.17977771934093825, -1.0285492007830714, 0.17752689532359736, -0.13484999105408077, 0.47573202290486277, 0.031822116699741179, 0.24037650281673009, 0.15590252477204442, 0.45738774129782067, 0.22173185702736309, 0.30187280005410128, -0.39975635000519244, -0.039503590406585444, -0.39817029220437605, -0.15445478010461899, -0.36489903195252488, -0.3853298858766015, -0.37582804923525442, -0.61443272417223993, -0.29601754827372728, -0.24418776554602573, -0.086406669789122897, -0.30224013030483832, -0.099140496022356861, -0.29063481045244749, -0.58087141114999497, 0.16985700712153248, -0.96642953239996532, -0.37729311110979141, -1.1701242444379498, -0.74333971957784439, 0.38028696922535249, -0.078120082418507292, -0.028162187795869011, -0.125541535197599, -0.047901768719509095, -0.072181687305083786, 0.11651314972109583, 0.23087406221635273, 0.13716928075053589, 0.060809740022726072, 0.20506236217269447, -0.67111321103900468, 0.094384329972762374, 0.15966636489766317, 0.19877934924652937, 0.11913422108293134, 0.25075678515412581, 0.074561341551142635, 0.14240747037292339, 0.12445548869448063, -0.00098732261430390365, 0.019759003776653159, 0.083680935776941467, 0.068718913680405266, 0.17488449732544983, 0.12383717136170486, 0.11152176458016351, 0.19769489878716068, 0.17362500400983244, 0.17111235393247615, 0.068205385795304344, 0.34985536178654059, 0.082741466859888216, 0.008743550726181536, 0.082154763578686424, 0.31068662937930014, 0.1518889379203984, -0.20242149238136126, 0.17888110216145156, 0.064195139226926753, 0.1213605232893029, 0.14152913847225079, 0.07106719540554117, 0.073300709749748075, 1.1109004319600426, 0.10895774920237174, 0.090760147222304077, 0.43188408735115758, -0.96642842630901471, -0.053928303635605149, 0.08021374160823519, -0.40210874775639521, -0.59481054551706036]

def read_train_data(train_data_path):
    x = []
    y = []
    with open(train_data_path[0], 'r') as X_train, \
         open(train_data_path[1], 'r') as Y_train:
            X_train.readline()
            Y_train.readline()
            for x_, y_ in zip(X_train, Y_train):
                lt = list(map(int, x_.strip().split(',')))
                x.append([1] + lt + list(np.array(lt[0:4])**2) + [lt[5]**2])
                y.append(int(y_))
    return x, y

def read_test_data(test_data_path):
    x_test = []
    with open(test_data_path, 'r') as X_test:
        X_test.readline()
        for x_ in X_test:
            lt = list(map(int, x_.strip().split(',')))
            x_test.append([1] + lt + list(np.array(lt[0:4])**2) + [lt[5]**2])
    return x_test

def feature_normalization(x):
    feature_mean = np.sum(x, axis=0) / len(x)
    feature_std = np.std(x, axis=0)
    feature_mean[0], feature_std[0] = 0, 1
    for i in range(len(x)):
        x[i] = (x[i] - feature_mean) / feature_std
    return x, feature_mean, feature_std

def test_feature_normalization(x, feature_mean, feature_std):
    for i in range(len(x)):
        x[i] = (x[i] - feature_mean) / feature_std
    return x
    
def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def J_theta(x, y, theta, regularization_lambda):
    total_cost = 0.
    for x_, y_ in zip(x, y):
        total_cost += cost(x_, y_, theta)
    regularization_term = np.power(theta, 2)[1:].sum() * regularization_lambda / (2 * len(x))
    return (total_cost / len(x)) + regularization_term

def cost(x_, y_, theta):
    if sigmoid(np.dot(x_, theta)) == 1:
        if y_ == 0:
            return 0
        else:
            return 0
    return -(y_ * np.log(sigmoid(np.dot(x_, theta)))) \
            - (1 - y_) * np.log(1. - sigmoid(np.dot(x_, theta)))

def logistic_regression(x, y, eta, regularization_lambda, epoch):
    feature_num = len(x[0])
    theta = np.random.randn(feature_num)
    J = J_theta(x, y, theta, regularization_lambda)
    for e in range(epoch):
        if e % 2000 == 0:
            print('Epoch: %d Loss: %f Acc: %f'% ((e+1), J_theta(x, y, theta, regularization_lambda), test_train_data(x, y, theta)))
        regularization_grad = (regularization_lambda / len(x)) * (theta**2)
        regularization_grad[0] = 0
        theta = (theta + regularization_grad) - eta * (1 / len(x)) * np.dot(np.array(x).transpose(), (sigmoid(np.dot(x, theta)) - y))
    return theta
    
def test_train_data(x, y, theta, threshold=0.5):
    correct = 0.
    for x_, y_ in zip(x, y):
        p = sigmoid(np.dot(x_, theta)) 
        if p >= threshold and y_ == 1:
            correct += 1
        elif p < threshold and y_ == 0:
            correct += 1
    return correct / float(len(x))
        
def test(x_test, theta, answer_path):
    if os.path.exists(answer_path):
        os.remove(answer_path)
    with open(answer_path, 'a') as ans:
        ans.write('id,label\r\n')
        id = 1
        for x_ in x_test:
            p = sigmoid(np.dot(x_, theta)) 
            if p >= 0.5:
                str ='%d,%d\r\n' % (id, 1)
                ans.write(str)
            elif p < 0.5:
                str ='%d,%d\r\n' % (id, 0)
                ans.write(str)
            id += 1

if __name__ == '__main__':

	x_train = sys.argv[3]
	y_train = sys.argv[4]
	train_data_path = [x_train, y_train]
	test_data_path = sys.argv[5]
	answer_path = sys.argv[6]
	x, y = read_train_data(train_data_path)
	x, feature_mean, feature_std = feature_normalization(x)
	x_test = read_test_data(test_data_path)
	x_test = test_feature_normalization(x_test, feature_mean, feature_std)
	test(x_test, theta, answer_path)