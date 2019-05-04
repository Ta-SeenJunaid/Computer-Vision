# -*- coding: utf-8 -*-

from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

import matplotlib.pyplot as plt
plt.imshow(x_train[20],cmap='gray_r')

y_train[20]

x_train = x_train/x_train.max()
x_test = x_test/x_train.max()

#reshape
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

from keras.utils import to_categorical

y_cat_train = to_categorical(y_train)
y_cat_test = to_categorical(y_test)

from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten

model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()

model.fit(x_train,y_cat_train,epochs=10)

model.save('fashion.h5')

model.evaluate(x_test,y_cat_test)

from sklearn.metrics import classification_report
predictions = model.predict_classes(x_test)

print(classification_report(y_test,predictions))









