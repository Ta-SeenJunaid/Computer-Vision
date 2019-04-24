# -*- coding: utf-8 -*-

from keras.datasets import cifar10
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

x_train = x_train/255
x_test = x_test/255

from keras.utils import to_categorical

y_cat_train = to_categorical(y_train,10)
y_cat_test = to_categorical(y_test,10)

from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten

model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(32,32,3),activation='relu'))




