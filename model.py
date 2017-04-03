import cv2
import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
from keras.models import Sequential,Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D,Input
from keras.layers.pooling import MaxPooling2D,AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras.applications.vgg16 import VGG16
from keras.regularizers import l2
import pydot
import graphviz
from keras.utils import plot_model
import random

import warnings
warnings.filterwarnings("ignore")


Steering_Angle=0.15

#Function to change the directory path
def changePath(st):
    return st.replace('/Users/vibhorsaxena/Desktop/driving_data/IMG','driving_data/IMG')
    
#Function to augment brightness    
def augment_brightness(image):
    im = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_brightness = random.uniform(0.5,1.2)
    im[:,:,2] = im[:,:,2]*random_brightness
    im = cv2.cvtColor(im,cv2.COLOR_HSV2RGB)
    return im

#Reading the data as a pandas dataframe and changing paths
df=pd.read_csv('driving_data/driving_log.csv', names=['Center', 'Left', 'Right', 'Angle', 'Throttle', 'Break','Speed'])
df['Left']= df['Left'].apply(changePath)
df['Right']= df['Right'].apply(changePath)
df['Center']= df['Center'].apply(changePath)
print (df.shape)

#Sampling and shuffling the data
df=df.sample(frac=0.65).reset_index(drop=True)
y=[]
X=np.empty(shape=[df.shape[0]*2,160,320,3])

#Distributing the data between center, left and right images in 60:20:20 ratio. Also applying flipping and brightness augmentation
print ("Starting Pre-Processing")
for index, row in df.iterrows():
    random_num=random.uniform(0,1)
    if (index%1000==0):
    	print ("Completed "+str(index)+" of "+str(df.shape[0])+" images.")
    if (random_num<0.6):
        im=cv2.imread(row['Center'])
        X[2*index]=augment_brightness(cv2.cvtColor(cv2.imread(row['Center']),cv2.COLOR_BGR2RGB))
        X[2*index]=augment_brightness(cv2.cvtColor(cv2.imread(row['Center']),cv2.COLOR_BGR2RGB))
        X[2*index+1]=augment_brightness(cv2.flip(cv2.cvtColor(cv2.imread(row['Center']),cv2.COLOR_BGR2RGB),1))
        y.append(row['Angle'])
        y.append(row['Angle']*-1.0)
    elif(random_num<0.80):
        X[2*index]=augment_brightness(cv2.cvtColor(cv2.imread(row['Left']),cv2.COLOR_BGR2RGB))
        X[2*index+1]=augment_brightness(cv2.flip(cv2.cvtColor(cv2.imread(row['Left']),cv2.COLOR_BGR2RGB),1))
        y.append(row['Angle']+Steering_Angle)
        y.append((row['Angle']+Steering_Angle)*-1.0)
    else:
        X[2*index]=augment_brightness(cv2.cvtColor(cv2.imread(row['Right']),cv2.COLOR_BGR2RGB))
        X[2*index+1]=augment_brightness(cv2.flip(cv2.cvtColor(cv2.imread(row['Right']),cv2.COLOR_BGR2RGB),1))
        y.append(row['Angle']-Steering_Angle)
        y.append((row['Angle']-Steering_Angle)*-1.0)

    
y=np.array(y)
shuffle(X, y)
    
#Creating the Nvidia Model    
def getNvidiaModel():
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 -0.5,input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(filters=24,kernel_size=(5,5),strides=(2,2),activation="relu"))
    model.add(Dropout(0.15))
    model.add(Convolution2D(filters=36,kernel_size=(5,5),strides=(2,2),activation="relu"))
    model.add(Dropout(0.15))
    model.add(Convolution2D(filters=48,kernel_size=(5,5),strides=(2,2),activation="relu"))
    model.add(Dropout(0.15))
    model.add(Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation="relu"))
    model.add(Dropout(0.15))
    model.add(Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation="relu"))
    model.add(Dropout(0.15))
    model.add(Flatten())
    model.add(Dense(1164, activation="relu"))
    model.add(Dropout(0.15))
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.15))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.15))
    model.add(Dense(10, activation="relu"))
    model.add(Dropout(0.15))
    model.add(Dense(1, activation="linear"))
    return model
        
#Creating the VGG Model  
def getVggModel():
    """Pretrained VGG16 model with fine-tunable last two layers"""
    input_image = Input(shape = (160,320,3))
    
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 -0.5,input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    
    base_model = VGG16(input_tensor=input_image, include_top=False)
        
    for layer in base_model.layers[:-3]:
        layer.trainable = False

    W_regularizer = l2(0.01)

    x = base_model.get_layer("block5_conv3").output
    x = AveragePooling2D((2, 2))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(4096, activation="elu", W_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation="elu", W_regularizer=l2(0.01))(x)
    x = Dense(2048, activation="elu", W_regularizer=l2(0.01))(x)
    x = Dense(1, activation="linear")(x)
    return Model(input=input_image, output=x)


#Training the nvidia model
model = getNvidiaModel()
model.compile(loss='mse',optimizer='adam')

#Splitting the training and validation data
model.fit(X,y,validation_split=Steering_Angle,shuffle=True,epochs=2)
model.save('model.h5')
plot_model(model, to_file='nvidia.png', show_shapes=True)


#########Code For Generator Below. This Was Left Unused Because The GPU Instance Could Handle The Data In-Memory Which Was Much Faster##########

'''def generator(df, batch_size=32):
    num_samples = df.shape[0]
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, int(batch_size/6)):
            batch_samples = df[offset:offset+batch_size].reset_index(drop=True)

            images = []
            angles = []
            y=[]
            X=np.empty(shape=[batch_samples.shape[0]*6,160,320,3])
            X=X.astype(np.float32)
            y=[]
            for index, row in batch_samples.iterrows():
                X[6*index]=augment_brightness(cv2.cvtColor(cv2.imread(row['Center']),cv2.COLOR_BGR2RGB))
                X[6*index+1]=augment_brightness(cv2.cvtColor(cv2.imread(row['Left']),cv2.COLOR_BGR2RGB))
                X[6*index+2]=augment_brightness(cv2.cvtColor(cv2.imread(row['Right']),cv2.COLOR_BGR2RGB))
                X[6*index+3]=augment_brightness(cv2.flip(cv2.cvtColor(cv2.imread(row['Center']),cv2.COLOR_BGR2RGB),1))
                X[6*index+4]=augment_brightness(cv2.flip(cv2.cvtColor(cv2.imread(row['Left']),cv2.COLOR_BGR2RGB),1))
                X[6*index+5]=augment_brightness(cv2.flip(cv2.cvtColor(cv2.imread(row['Right']),cv2.COLOR_BGR2RGB),1))
                y.append(row['Angle'])
                y.append(row['Angle']+Steering_Angle)
                y.append(row['Angle']-Steering_Angle)
                y.append(row['Angle']*-1.0)
                y.append((row['Angle']+Steering_Angle)*-1.0)
                y.append((row['Angle']-Steering_Angle)*-1.0)
                
            y=np.array(y)
            yield shuffle(X, y)

from sklearn.model_selection import train_test_split
df=pd.read_csv('driving_data/driving_log.csv', names=['Center', 'Left', 'Right', 'Angle', 'Throttle', 'Break','Speed'])
df['Left']= df['Left'].apply(changePath)
df['Right']= df['Right'].apply(changePath)
df['Center']= df['Center'].apply(changePath)

df=df.sample(frac=0.75).reset_index(drop=True)

train_samples, validation_samples = train_test_split(df, test_size=0.1)
train_generator = generator(train_samples, batch_size=600)
validation_generator = generator(validation_samples, batch_size=600)

model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)'''