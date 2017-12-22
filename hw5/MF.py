
# coding: utf-8

# In[144]:


import os
import json
import numpy as np
import matplotlib.colors
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from keras.layers import Embedding, Reshape, Dot, Merge, Add
from keras.layers import Dropout, Dense, Input, Flatten, Concatenate
from keras.models import Sequential, load_model, Model
from keras.models import model_from_json
from keras.regularizers import l2
get_ipython().run_line_magic('matplotlib', 'inline')


# In[342]:


train_data_path = '/home/csdai0324/hw5_data/train.csv'
user_data_path = '/home/csdai0324/hw5_data/users.csv'
test_data_path = '/home/csdai0324/hw5_data/test.csv'
answer_path = '/home/csdai0324/hw5_data/answer.txt'


# In[389]:


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

def read_users(user_data_path, max_num):
    with open(user_data_path, 'r') as user_data:
        user_data.readline()
        gender = ['A']*(max_num + 1)
        age = [0]*(max_num + 1)
        for line in user_data:
            lt = line.strip().split('::')
            gender[int(lt[0])] = lt[1]
            age[int(lt[0])] = int(lt[2])
    return gender, age

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
    model = model_from_json(model_json)
    model.load_weights(model_weight_path)
    return model

def predict_rating(userid, movieid, model):
    return model.predict([np.array([userid]), np.array([movieid])])[0][0]

def age_gender_predict_rating(userid, movieid, model, age, gender):
    return model.predict([np.array([userid]), np.array([movieid]), np.array([age]), np.array([gender])])[0][0]
    
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

def age_gender_predict(answer_path, model, test_data_path, input_age, input_gender):
    if os.path.isfile(answer_path):
        os.remove(answer_path)
    test_users, test_movies = read_test_data(test_data_path)
    pred = []
    for i in range(len(test_users)):
        pred.append(age_gender_predict_rating(test_users[i], test_movies[i], model, input_age[i], input_gender[i]))
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


# In[386]:


users, movies, ratings = read_train_data(train_data_path)
gender, age = read_users(user_data_path, max(users))
input_gender = []
input_age = []
for user in users:
    if gender[user] == 'M':
        input_gender.append(1)
    else:
        input_gender.append(0)
    input_age.append(age[user])
#ratings, mean, std = standard_normalization(ratings)


# In[369]:


def z_model(user_num, movie_num, factors):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])
    user_vec = Embedding(user_num, factors, embeddings_initializer='random_normal')(user_input)
    user_vec = Flatten()(user_vec)
    item_vec = Embedding(movie_num, factors, embeddings_initializer='random_normal')(item_input)
    item_vec = Flatten()(item_vec)
    user_bias = Embedding(user_num, 1, embeddings_initializer='zeros')(user_input)
    user_bias = Flatten()(user_bias)
    item_bias = Embedding(movie_num, 1, embeddings_initializer='zeros')(item_input)
    item_bias = Flatten()(item_bias)
    r_hat = Dot(axes=1)([user_vec, item_vec])
    r_hat = Add()([r_hat, user_bias, item_bias])
    model = Model([user_input, item_input], r_hat)
    model.compile(loss='mse', optimizer='adamax')
    model.summary()
    return model

def z_dnn_model(user_num, movie_num, factors):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])
    age_input = Input(shape=[1])
    gender_input = Input(shape=[1])
    user_vec = Embedding(user_num, factors, embeddings_initializer='random_normal')(user_input)
    user_vec = Flatten()(user_vec)
    item_vec = Embedding(movie_num, factors, embeddings_initializer='random_normal')(item_input)
    item_vec = Flatten()(item_vec)
    merge_vec = Concatenate()([user_vec, item_vec])
    merge_vec = Concatenate()([merge_vec, age_input])
    merge_vec = Concatenate()([merge_vec, gender_input])
    hidden = Dense(256, activation='relu')(merge_vec)
    hidden = Dropout(0.2)(hidden)
    hidden = Dense(128, activation='relu')(hidden)
    hidden = Dropout(0.2)(hidden)
    output = Dense(1, activation='linear')(hidden)
    model = Model([user_input, item_input, age_input, gender_input], output)
    model.compile(loss='mse', optimizer='adamax')
    model.summary()
    return model


# In[370]:


model = z_dnn_model(max(users), max(movies), 256)
model.fit([np.array(users), np.array(movies), np.array(input_age), np.array(input_gender)], np.array(ratings), batch_size=512, epochs=10, verbose=1)


# In[373]:


model.fit([np.array(users), np.array(movies), np.array(input_age), np.array(input_gender)], np.array(ratings), batch_size=512, epochs=5, verbose=1)


# In[387]:


test_users, test_movies = read_test_data(test_data_path)
input_gender = []
input_age = []
for user in test_users:
    if gender[user] == 'M':
        input_gender.append(1)
    else:
        input_gender.append(0)
    input_age.append(age[user])


# In[390]:


#pred = nor_predict(answer_path, model, test_data_path, mean, std)
pred = age_gender_predict(answer_path, model, test_data_path, input_age, input_gender)


# In[86]:


model_json_path = '/home/csdai0324/ML2017FALL/hw5/model/model_json.txt'
model_weigth_path = '/home/csdai0324/ML2017FALL/hw5/model/model.h5'
model1 = load_model(model_json_path, model_weigth_path)
pred = predict(answer_path, model1, test_data_path)


# In[187]:


movie_embedding = model.get_weights()[2]


# In[119]:


with open('/home/csdai0324/hw5_data/movies.csv', 'r', encoding = "ISO-8859-1") as movie_csv:
    movie_csv.readline()
    c = Counter()
    for line in movie_csv:
        categories = line.strip().split('::')[2]
        categories_lt = categories.split('|')
        for category in categories_lt:
            c[category] += 1
    print(c)


# In[280]:


movie_emb = np.array(model.layers[3].get_weights()).squeeze()
print(movie_emb.shape)


# In[281]:


categories_used = ['Drama'    ,  'Comedy',      'Action', 
                   'Thriller' ,  'Horror',     'Romance', 
                   'Adventure',  'Sci-Fi',  "Children's",
                   'Crime'    ,     'War', 'Documentary',
                   'Musical'  , 'Mystery',   'Animation',
                   'Fantasy'  , 'Western',   'Film-Noir']
#categories_used = ['Animation', 'Musical', 'Horror', 'Thriller']


# In[282]:


y = []
x = []
counter = 0
with open('/home/csdai0324/hw5_data/movies.csv', 'r', encoding = "ISO-8859-1") as movie_csv:
    movie_csv.readline()
    for line in movie_csv:
        categories = line.strip().split('::')[2]
        categories_lt = categories.split('|')
        index = []
        flag = False
        for category in categories_lt:
            if category in categories_used:
                index.append(categories_used.index(category))
                flag = True
        if flag == True:
            if (14 in index):
                x.append(movie_emb[counter])
                y.append(100)
            elif (1 in index):
                x.append(movie_emb[counter])
                y.append(50)
            elif (3 in index) or (4 in index):
                x.append(movie_emb[counter])
                y.append(10)
        counter += 1       


# In[283]:


def tsne(x, y):
    x = np.array(x, dtype=np.float64)
    y = np.array(y)
    vis_data = TSNE(n_components=2).fit_transform(x)
    return vis_data

vis_data = tsne(x, y)


# In[287]:


print(vis_x[:5])
print(vis_y[:5])
print(y[:5])


# In[341]:


def draw(vis_data, y):
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]
    cm = plt.cm.get_cmap('winter')
    sc = plt.scatter(vis_x, vis_y, cmap=cm)
    plt.colorbar(sc)
    plt.show()
    
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]
area = np.pi * (3)**2 
Animation = []
Thriller_Horror = []
Comedy = []
Animation_y = []
Thriller_Horror_y = []
Comedy_y = []
for i in range(len(vis_x)):
    if y[i] == 100:
        Animation.append(vis_x[i])
        Animation_y.append(vis_y[i])
    elif y[i] == 50:
        Comedy.append(vis_x[i])
        Comedy_y.append(vis_y[i])
    elif y[i] == 10:
        Thriller_Horror.append(vis_x[i])
        Thriller_Horror_y.append(vis_y[i])
#plt.scatter(Animation, Animation_y, c=[100]*len(Animation), s=area, cmap='viridis', label='Animation') 
plt.scatter(vis_x, vis_y, c=y, s=area, cmap='viridis')
#plt.scatter(Comedy, Comedy_y, c=[50]*len(Comedy), s=area, cmap='viridis', label='Comedy') 
#plt.scatter(Thriller_Horror, Thriller_Horror_y, c=[0]*len(Thriller_Horror), s=area, cmap='viridis', label='Thriller & Horror') 
#plt.legend()
plt.show()
#draw(vis_data, y)

