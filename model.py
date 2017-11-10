import csv
import cv2
import numpy as np
import gc

import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout, Lambda, Cropping2D
from keras import optimizers

# Training parameters
epochs = 7
learning_rate = 0.0001
batch_size = 256

samples = []
with open('./raw_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Shuffle all loaded data before split to training and validation data-sets
sklearn.utils.shuffle(samples)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print("samples.shape: ", len(samples))
print("train_samples.shape: ", len(train_samples))
print("validation_samples.shape: ", len(validation_samples))

def generator(samples, batch_size):
    shuffle(samples)
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                flag = random.randint(0, 2)  # Draw an int random number between 0 and 2
                if flag == 0:  # Center
                    image = cv2.imread(batch_sample[0].strip())
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    angle = float(line[3])
                    images.append(image)
                    angles.append(angle)
                elif flag == 1:  # Left
                    image = cv2.imread(batch_sample[1].strip())
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    angle = float(line[3]) + 0.25 # Correct the driving angle
                    images.append(image)
                    angles.append(angle)
                elif flag == 2:  # Right
                    image = cv2.imread(batch_sample[2].strip())
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    angle = float(line[3]) - 0.25 # Correct the driving angle
                    images.append(image)
                    angles.append(angle)
                else:
                    print("Error on Center/Left/Right random image draw!")

        X_train = np.array(images)
        y_train = np.array(angles)
        yield sklearn.utils.shuffle(X_train, y_train)

# Model
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))
model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='same', activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

Adam = optimizers.Adam(lr=learning_rate)

model.compile(loss='mse', optimizer=Adam)

history_object = model.fit_generator(generator(train_samples, batch_size),
                                     steps_per_epoch=int(len(train_samples)/batch_size),
                                     validation_data=generator(validation_samples, batch_size),
                                     validation_steps=int(len(validation_samples)/batch_size),
                                     epochs=epochs)

model.save('model.h5')

gc.collect()  # Garbage collector

# Print the keys contained in the history object
print(history_object.history.keys())

# Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss.png')