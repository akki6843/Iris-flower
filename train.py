import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from matplotlib import pyplot as plt


data = pd.read_csv("iris.csv")
SL =data['SepalLengthCm']
SW = data['SepalWidthCm']
PL = data['PetalLengthCm']
PW = data['PetalWidthCm']

X_train = pd.concat([SL, SW, PL, PW], axis=1)
print(X_train.head())
y = data['labels']
Y_train = to_categorical(y, num_classes=3)
def create_model():
    model = Sequential()
    model.add(Dense(500, activation='relu', input_shape=(4,)))
    model.add(Dense(250,activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(3,activation='softmax'))
    return model

model = create_model()
model.summary()

def train():
    model = create_model()
    model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=8, epochs=10, verbose=1,shuffle=True, validation_split=0.1)
    model.save_weights('./model/iris.h5')



train()