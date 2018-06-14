# -*- coding: utf-8 -*-
__author__ = 'Jinkey'

from Data import mnist
import numpy as np
from keras import *
print(np.array(mnist.train_images).shape)


model = Sequential()
model.add(layers.Conv2D(filters=6, input_shape=(227, 227, 3), kernel_size=(5, 5), strides=1))
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=2))
model.add(layers.Activation('sigmoid'))
model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), strides=1))
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=2))
model.add(layers.Activation('sigmoid'))
model.add(layers.Flatten())
model.add(layers.Dense(120, activation='tanh'))
model.add(layers.Dense(84, activation='tanh'))
model.add(layers.Dense(10, activation='softmax'))

sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(np.expand_dims(mnist.train_images, axis=3), utils.to_categorical(mnist.train_labels), batch_size=500, epochs=20, verbose=1, shuffle=True)

score = model.evaluate(np.expand_dims(mnist.test_images, axis=3), utils.to_categorical(mnist.test_labels))
print("Total loss on Testing Set:", score[0])  # 0.049
print("Accuracy of Testing Set:", score[1])   # 98.43%
