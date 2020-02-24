import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl
from collections import defaultdict
from sklearn.model_selection import train_test_split





data = pd.read_csv("train.csv")
data1 = pd.read_csv('test.csv')

y = data['Death_Count']
del data['Death_Count']
print(data)
mean_value=data['Disaster'].mode()
data['Disaster']=data['Disaster'].fillna(mean_value)


mean_value=data['Planet_ID'].mode()
data['Planet_ID']=data['Planet_ID'].fillna(mean_value)
X_val, X_test, Y_val, Y_test = train_test_split(data, y, test_size=0.5)
from keras.models import Sequential
from keras.layers import Dense
model = Sequential([
    Dense(32, activation='relu', input_shape=(5,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])
model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])
hist = model.fit(X_test, Y_test,
          batch_size=32, epochs=100,
          validation_data=(X_val, Y_val))

























'''
import csv
with open ('output.csv' , 'w' , newline = '') as file:
    writer = csv.writer(file , delimiter=',')
    writer.writerow(['Accident_ID','Severity'])
    for i in res:
        writer.writerow(i)
'''
