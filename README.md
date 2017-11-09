# Behavioral Cloning
Behavioral Cloning Project - Udacity Self-Driving Car Engineer Nanodegree. 

## Description ##
This project is a part of the Udacity's Self Driving Car Nanodegree. The main focus is to develop a deep neural network capable to drive a vehicle on a simulated track.
The input data is generated from human behavior using the same simulator.

## Deliverables ##

`model.py` - Main script containing neural network.

`datasetProcessor.py` - Dataset preparation and training data augmentation.

`model.h5` - Trained model that is able to complete successfully a track. Please, download it from [here](https://1drv.ms/u/s!AqUS_0zt3Km9iDR9_JqVahtRuxP1) (1.32 GB).

`run1.mp4` - The video file containing 2 successfull laps completed in an automonuous driving mode. Please, download it from [here](https://1drv.ms/v/s!AqUS_0zt3Km9iDNzuRL8rA9Ur-vG) (43.9 MB, 320 X 160).

## Setup ##
The environment can be setup following this link: [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit/)

More detailed project description is available [here](https://github.com/udacity/CarND-Behavioral-Cloning-P3). 

Training data can be downloaded from [here](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip). It is important to remove a first descriptive line in a *csv* file.

## Implementation ##
The neural network architecture used in this project follows, in general terms, the Nvidia's DNN described in [this](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) blog post. The simplified neural network was enough to get a track completed. Network's implementation using [Keras](https://keras.io) is shown below:
```
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
```
The Adam optimizer with learning rate 0.0001 was used.
```
Adam = optimizers.Adam(lr=learning_rate)
model.compile(loss='mse', optimizer=Adam)
```
The code above was transformed to the following:
```
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 45, 160, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 23, 80, 48)        28848     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 23, 80, 64)        27712     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 23, 80, 64)        36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 117760)            0         
_________________________________________________________________
dense_1 (Dense)              (None, 1000)              117761000 
_________________________________________________________________
dropout_1 (Dropout)          (None, 1000)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 100)               100100    
_________________________________________________________________
dropout_2 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 50)                5050      
_________________________________________________________________
dropout_3 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 11        
=================================================================
Total params: 117,961,983
Trainable params: 117,961,983
Non-trainable params: 0
_________________________________________________________________

```

At the training stage, the neural network above is feeded with a data by generator. The generator function selects randomly central, left, or right picture from one string of the training dataset. All training data augmentation is done by `datasetProcessor.py`.

The training dataset is composed by the following pictures:
- original pictures
- pictures flipped horizontally
- original pictures with random brightness
- flipped pictures with random brightness

Training was performed with the following parameters:
- epochs = 7
- learning_rate = 0.0001
- batch_size = 256

Training and validation performance of the model is depicted in a Figure below.
![loss](https://github.com/antonpavlov/behavioral-cloning/blob/master/loss.png)


## License ##
Files `model.py`, `model.h5`, `datasetProcessor.py`, and `run1.mp4` are distributed under MIT license.
Please refer to [Udacity](https://www.udacity.com/) regarding all other supporting materials.