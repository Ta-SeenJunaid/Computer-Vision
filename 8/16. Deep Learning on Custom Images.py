import matplotlib.pyplot as plt
import cv2

cat4 = cv2.imread('../DATA/CATS_DOGS/train/cats/cat.4.jpg')
cat4 = cv2.cvtColor(cat4,cv2.COLOR_BGR2RGB)
plt.imshow(cat4)
cat4.shape

dog2 = cv2.imread('../DATA/CATS_DOGS/train/dogs/dog.2.jpg')
dog2 = cv2.cvtColor(dog2,cv2.COLOR_BGR2RGB)
plt.imshow(dog2)
dog2.shape

from keras.preprocessing.image import ImageDataGenerator
image_gen = ImageDataGenerator(rotation_range=40,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               rescale=1/255,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               fill_mode='nearest'
                               )

plt.imshow(image_gen.random_transform(dog2))

image_gen.flow_from_directory('../DATA/CATS_DOGS/train')

from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Dense

model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(150,150,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=(150,150,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=(150,150,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units= 128,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(units= 1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

train_image_gen = image_gen.flow_from_directory('../DATA/CATS_DOGS/train',
                                                target_size=(150,150),
                                                batch_size=16,
                                                class_mode='binary')

test_image_gen = image_gen.flow_from_directory('../DATA/CATS_DOGS/test',
                                                target_size=(150,150),
                                                batch_size=16,
                                                class_mode='binary')

train_image_gen.class_indices

results = model.fit_generator(train_image_gen,epochs=100,
                              steps_per_epoch=150,
                              validation_data=test_image_gen,
                              validation_steps=12)

model.save('CATS_DOGS.h5')

plt.plot(results.history['acc'])

#for multiple classification binary>categorical, sigmoid>softmax


#Making new predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('../DATA/CATS_DOGS/single_prediction/cat_or_dog_1.jpg', target_size = (150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
train_image_gen.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'


#Making new predictions with saved model
from keras.models import load_model
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('../DATA/CATS_DOGS/single_prediction/cat_or_dog_2.jpg', target_size = (150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

newmodel = load_model('CATS_DOGS.h5')
result = newmodel.predict_classes(test_image)


if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'