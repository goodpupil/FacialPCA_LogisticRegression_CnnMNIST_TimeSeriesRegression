import pandas as pd
import numpy as np
# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

import sys


train_array = np.genfromtxt(sys.argv[1], delimiter=';', skip_header=1)
train_array = train_array[:,2]
mean = np.nanmean(train_array)

no_of_samples = 50000
counter=0
samp_array = np.array([0]*61)
while counter < no_of_samples:
#     print(counter)
    index = np.random.randint(low=0, high=len(train_array)-60-1)
    slice_ = train_array[index:index+61]
    if np.any(np.isnan(slice_)) == False:
        samp_array = np.vstack((samp_array, slice_))
        counter += 1
samp_array = samp_array[1:,:].copy()

x_train, y_train = samp_array[:,:-1], samp_array[:,-1]

reg = LinearRegression()
reg.fit(x_train, y_train)

pred = -1
counter=0
for ti in range(len(train_array)):
    if np.isnan(train_array[ti]):
        if ti < 60:
            pred = np.nanmean()
        else:
            x_test = train_array[ti-60:ti]
            pred = reg.predict(x_test.reshape(1,-1))
        print(pred[0])
        counter += 1
        train_array[ti] = pred
# print(counter)