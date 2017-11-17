import csv
import sys
import os
import numpy as np
import keras

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Add, Input, Multiply, Concatenate
from keras.layers.normalization import BatchNormalization

def read_train_data(train_data_path):
    x = []
    y = []
    with open(train_data_path, 'r') as train_data:
        train_data.readline()
        for line in train_data:
            label, pixels = line.strip().split(',')
            pixels = list(map(int, pixels.split(' ')))
            x.append(pixels)
            y.append(int(label))
    return x, y

def my_model(input_shape, num_classes):
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=input_shape, padding='same'))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.3))
	model.add(Conv2D(32, (3, 3), activation='linear', padding='same'))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.3))
	model.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, kernel_size=(3, 3), activation='linear', input_shape=input_shape, padding='same'))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.3))
	model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.3))
	model.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, kernel_size=(3, 3), activation='linear', input_shape=input_shape, padding='same'))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.3))
	model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.3))
	model.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))
	model.add(Dropout(0.2))
	model.add(Conv2D(128, kernel_size=(3, 3), activation='linear', input_shape=input_shape, padding='same'))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.3))
	model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.3))
	model.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))
	model.add(Conv2D(128, kernel_size=(3, 3), activation='linear', input_shape=input_shape, padding='same'))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.3))
	model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.3))
	model.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(4096, activation='linear'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(LeakyReLU(alpha=0.3))
	model.add(Dense(2048, activation='linear'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(LeakyReLU(alpha=0.3))
	model.add(Dense(1024, activation='linear'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(LeakyReLU(alpha=0.3))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
	              metrics=['accuracy'])

	return model

def train(model, x_train, y_train, batch_size):
	datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, 
                             height_shift_range=0.2, shear_range=0.2, 
                             zoom_range=0.2, horizontal_flip=True,
                             fill_mode='nearest')
	datagen.fit(x_train)
	model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
	                    steps_per_epoch=len(x_train) / 128, verbose=1, epochs=250)

def main():
	train_data_path = sys.argv[1]
	batch_size = 128
	num_classes = 7
	epochs = 50
	img_rows, img_cols = 48, 48
	input_shape = (48, 48, 1)
	x_train, y_train = read_train_data(train_data_path)
	x_train = np.array(x_train).reshape(np.array(x_train).shape[0], img_rows, img_cols, 1)
	x_train = x_train.astype('float32')
	x_train /= 255
	y_train = keras.utils.to_categorical(y_train, num_classes)
	train(my_model(input_shape, num_classes), x_train, y_train, batch_size)
	model.save('model.h5')	

if __name__ == '__main__':
	main()



