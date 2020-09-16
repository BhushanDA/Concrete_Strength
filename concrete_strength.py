# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 13:54:01 2020

@author: Administrator
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as skl_mtc
from sklearn.model_selection import train_test_split

df = pd.read_csv("D:\Bhushan\concrete_data.csv")
df.head()
df.isnull()

axs = pd.plotting.scatter_matrix(df, figsize=(15, 15))
corr=sns.heatmap(df.corr(),annot=True)

df1= df.drop("Strength",axis=1)
tf.keras.utils.normalize(df1, axis=1)


pred=df['Strength'] 
pred.head()
pred1=pred.to_frame()

X_train, X_test, y_train, y_test = train_test_split(df1, pred1, test_size=0.33, random_state=42

model = Sequential()
model.add(keras.layers.Dense(6000, activation='relu', input_dim=8))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(200, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(1, kernel_initializer='uniform'))
model.summary()

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

history = model.fit(
    X_train,
    y_train,
    epochs=200,
    validation_data=(X_test,y_test))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


p=model.predict(X_test)
result = skl_mtc.r2_score(y_test, p)
y_test.head()
