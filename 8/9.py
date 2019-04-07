# -*- coding: utf-8 -*-
import numpy as np
from numpy import genfromtxt

data = genfromtxt('../DATA/bank_note_data.txt',delimiter=',')
data


labels = data[:,4]
features = data[:,0:4]

X=features
y=labels

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler_object=MinMaxScaler()
scaler_object.fit(X_train)

scaled_X_train = scaler_object.transform(X_train)
scaled_X_test = scaler_object.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(4,input_dim=4,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(scaled_X_train,y_train,epochs=50,verbose=2)

#model.metrics_names
from sklearn.metrics import confusion_matrix,classification_report
predictions = model.predict_classes(scaled_X_test)
confusion_matrix(y_test,predictions)
print(classification_report(y_test,predictions))

model.save('mysupermodel.h5')

from keras.models import load_model
newmodel = load_model('mysupermodel.h5')
newmodel.predict_classes(scaled_X_test)






