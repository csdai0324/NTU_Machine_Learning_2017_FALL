from __future__ import print_function

from scipy.misc import imsave
import numpy as np
import time
from keras import backend as K
import keras 
K.set_learning_phase(1)
img_width = 48
img_height = 48
layer_name = 'conv2d_12'
train_data_path = './train.csv'

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
x_train, y_train = read_train_data(train_data_path)
x_train = np.array(x_train).reshape(np.array(x_train).shape[0], img_width, img_height, 1)
x_train = x_train.astype('float32')
x_train /= 255

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

model = keras.models.load_model('two.h5')
print('Model loaded.')

model.summary()
input_img = model.input
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])


def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

kept_filters = []
for filter_index in range(32):
    print('Processing filter %d' % filter_index)
    start_time = time.time()

    layer_output = layer_dict[layer_name].output
    if K.image_data_format() == 'channels_first':
        loss = K.mean(layer_output[:, filter_index, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, input_img)[0]

    grads = normalize(grads)

    iterate = K.function([input_img], [loss, grads])
    step = 1.
    if K.image_data_format() == 'channels_first':
        input_img_data = np.random.random((1, 1, img_width, img_height))
    else:
        input_img_data = np.random.random((1, img_width, img_height, 1))
    input_img_data = np.array(x_train[20]).reshape(1, img_width, img_height, 1)

    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        print('Current loss value:', loss_value)
        if loss_value <= 0.:
            break
    if loss_value < 0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
    end_time = time.time()
    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

n = 4

kept_filters.sort(key=lambda x: x[1], reverse=True)
kept_filters = kept_filters[:n * n]

margin = 5
width = n * img_width + (n - 1) * margin
height = n * img_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))
print(len(kept_filters))
for i in range(n):
    for j in range(n):
        img, loss = kept_filters[i * n + j]
        stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                         (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

imsave('stitched_filters_%dx%d.png' % (n, n), stitched_filters)