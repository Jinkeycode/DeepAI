# -*- coding: utf-8 -*-
__author__ = 'Jinkey'

from keras import *
from skimage import transform, color
from PIL import Image
from Data import mnist
import numpy as np


def images_tranform_gray28_to_rgb227(images: list):
    images_resize = []
    for index, image in enumerate(images[0:5000]):
        if index % 1000 == 0:
            print("正在处理 %s 张图片" %index)
        # image_resize = color.gray2rgb(transform.resize(image, (227, 227)))
            image_resize = transform.resize(image, (227, 227))
        images_resize.append(image_resize)

    return np.expand_dims(np.array(images_resize, dtype='float32'), axis=3)


# print(images_tranform_gray28_to_rgb227(mnist.train_images).shape)

model = Sequential()
model.add(layers.Conv2D(filters=96, kernel_size=(11, 11), strides=4, input_shape=(227, 227, 1)))
model.add(layers.MaxPool2D(pool_size=(3, 3), strides=2))
model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), padding='same'))
model.add(layers.MaxPool2D(pool_size=(3, 3), strides=2))
model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same'))
model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same'))
model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
model.add(layers.MaxPool2D(pool_size=(3, 3), strides=2))
model.add(layers.Flatten())
model.add(layers.Dense(4096))
model.add(layers.Dense(4096))
model.add(layers.Dense(1000, activation='softmax'))
model.add(layers.Dense(10, activation='softmax'))

sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(images_tranform_gray28_to_rgb227(mnist.train_images), utils.to_categorical(mnist.train_labels[0: 5000]), batch_size=500, epochs=5, verbose=1, shuffle=True)

# score = model.evaluate(images_tranform_gray28_to_rgb227(mnist.test_images), utils.to_categorical(mnist.test_labels))
# print("Total loss on Testing Set:", score[0])  # 0.049
# print("Accuracy of Testing Set:", score[1])   # 98.43%

