#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 10:18:51 2018

@author: natnaelhamda
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import pyplot

import keras
from keras import optimizers
import math
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
    
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from numba import jit

import timeit 

start = timeit.default_timer()

# Importing the training set
dataset_tran_test_val_in = pd.read_csv('data_in_out_cascade.csv') #all inputs 2000 - 2017
dataset_tran_test_val_in = np.array(dataset_tran_test_val_in)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1)) #For inputs
data_all_2000_2018_in_scaled = sc.fit_transform(dataset_tran_test_val_in)


# Predictors SHASTA PROFILE
training_set_scaled_SHA_P = data_all_2000_2018_in_scaled[:5275,[1,2,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]]

testing_set_scaled_SHA_P = data_all_2000_2018_in_scaled[5275:6207,[1,2,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]]

validation_set_scaled_SHA_P = data_all_2000_2018_in_scaled[6207:-1,[1,2,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]]

# Predictors SHASTA OUTFLOW Tempreature
training_set_scaled_SHA_T = data_all_2000_2018_in_scaled[:5275,[2,15,16,17,18,26,27,28]]

testing_set_scaled_SHA_T = data_all_2000_2018_in_scaled[5275:6207,[2,15,16,17,18,26,27,28]]

validation_set_scaled_SHA_T = data_all_2000_2018_in_scaled[6207:-1,[2,15,16,17,18,26,27,28]]

# Predictors Keswick   Tempreature
training_set_scaled_KWK_T = data_all_2000_2018_in_scaled[:5275,[2,3,4,20,21,22,23,24,25,29]]

testing_set_scaled_KWK_T = data_all_2000_2018_in_scaled[5275:6207,[2,3,4,20,21,22,23,24,25,29]]

validation_set_scaled_KWK_T = data_all_2000_2018_in_scaled[6207:-1,[2,3,4,20,21,22,23,24,25,29]]

## Response SHASTA PROFILE
training_set_scaled_resp_SHA_P = data_all_2000_2018_in_scaled[1:5276,[26, 27,28]]

testing_set_scaled_resp_SHA_P = data_all_2000_2018_in_scaled[5276:6208,[26, 27,28]]

validation_set_scaled_resp_SHA_P = data_all_2000_2018_in_scaled[6208:,[26, 27,28]]

## Response SHASTA OUTFLOW Tempreature
training_set_scaled_resp_SHA_T = data_all_2000_2018_in_scaled[1:5276,29]

testing_set_scaled_resp_SHA_T = data_all_2000_2018_in_scaled[5276:6208,29]

validation_set_scaled_resp_SHA_T = data_all_2000_2018_in_scaled[6208:,29]


## Response Keswick OUTFLOW Tempreature
training_set_scaled_resp_KWK_T = data_all_2000_2018_in_scaled[1:5276,30]

testing_set_scaled_resp_KWK_T = data_all_2000_2018_in_scaled[5276:6208,30]

validation_set_scaled_resp_KWK_T = data_all_2000_2018_in_scaled[6208:,30]

###############################################################################

###############################################################################

lag_day = 1


X_train_SHA_P = []
y_train_SHA_P = []

X_train_SHA_T = []
y_train_SHA_T = []

X_train_KWK_T = []
y_train_KWK_T = []

for i in range(lag_day, len(training_set_scaled_SHA_T)):
    X_train_SHA_P.append(training_set_scaled_SHA_P[i-lag_day:i, :]) #Assuming 5 days lag time affects, here discharge temperature not included  
    y_train_SHA_P.append(training_set_scaled_resp_SHA_P[i]) #Shashta
    
    X_train_SHA_T.append(training_set_scaled_SHA_T[i-lag_day:i, :]) #Assuming 5 days lag time affects, here discharge temperature not included  
    y_train_SHA_T.append(training_set_scaled_resp_SHA_T[i]) #Shashta
    
    X_train_KWK_T.append(training_set_scaled_KWK_T[i-lag_day:i, :]) #Assuming 5 days lag time affects, here discharge temperature not included  
    y_train_KWK_T.append(training_set_scaled_resp_KWK_T[i]) #Shashta

X_train_SHA_P, y_train_SHA_P = np.array(X_train_SHA_P), np.array(y_train_SHA_P)

X_train_SHA_T, y_train_SHA_T = np.array(X_train_SHA_T), np.array(y_train_SHA_T)

X_train_KWK_T, y_train_KWK_T = np.array(X_train_KWK_T), np.array(y_train_KWK_T)


# Reshaping
X_train_SHA_P = np.reshape(X_train_SHA_P, (X_train_SHA_P.shape[0], X_train_SHA_P.shape[1], 23))

X_train_SHA_T = np.reshape(X_train_SHA_T, (X_train_SHA_T.shape[0], X_train_SHA_T.shape[1], 8))

X_train_KWK_T = np.reshape(X_train_KWK_T, (X_train_KWK_T.shape[0], X_train_KWK_T.shape[1], 10))

#test_set_scaled = testing_set_scaled_resp_shashta

X_test_SHA_P = []
y_test_SHA_P = []

X_test_SHA_T = []
y_test_SHA_T = []

X_test_KWK_T = []
y_test_KWK_T = []

for i in range(lag_day, len(testing_set_scaled_SHA_T)):
    X_test_SHA_P.append(testing_set_scaled_SHA_P[i-lag_day:i, :])
    y_test_SHA_P.append(testing_set_scaled_resp_SHA_P[i])
    
    X_test_SHA_T.append(testing_set_scaled_SHA_T[i-lag_day:i, :])
    y_test_SHA_T.append(testing_set_scaled_resp_SHA_T[i])
    
    X_test_KWK_T.append(testing_set_scaled_KWK_T[i-lag_day:i, :])
    y_test_KWK_T.append(testing_set_scaled_resp_KWK_T[i])


X_test_SHA_P, y_test_SHA_P = np.array(X_test_SHA_P), np.array(y_test_SHA_P)

X_test_SHA_T, y_test_SHA_T = np.array(X_test_SHA_T), np.array(y_test_SHA_T)

X_test_KWK_T, y_test_KWK_T = np.array(X_test_KWK_T), np.array(y_test_KWK_T)


# Reshaping for keras formating
X_test_SHA_P = np.reshape(X_test_SHA_P, (X_test_SHA_P.shape[0], X_test_SHA_P.shape[1], 23))

X_test_SHA_T = np.reshape(X_test_SHA_T, (X_test_SHA_T.shape[0], X_test_SHA_T.shape[1], 8))

X_test_KWK_T = np.reshape(X_test_KWK_T, (X_test_KWK_T.shape[0], X_test_KWK_T.shape[1], 10))



###############################################################################

@jit 
def fit_model_SHA_P(X_train, y_train, X_test, y_test):
    # define model
    model_SHA_P = Sequential()
    model_SHA_P.add(LSTM(units = 23, return_sequences = True, input_shape = (X_train.shape[1], 23)))
    model_SHA_P.add(Dropout(0.2))
    
    model_SHA_P.add(LSTM(units = 23, return_sequences = True))
    model_SHA_P.add(Dropout(0.2))
    
    model_SHA_P.add(LSTM(units = 23))
    model_SHA_P.add(Dropout(0.2))
    
    model_SHA_P.add(Dense(units = 3))
    
    
    
    model_SHA_P.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    history = model_SHA_P.fit(X_train, y_train, epochs = 10, batch_size = 128, validation_data = (X_test, y_test),verbose=0)
    
    loss_SHA_P_train = history.history['loss'] #model_KWK_T.evaluate(X_test, y_test,verbose=0)
    
    loss_SHA_P_test = history.history['val_loss']
    
    return loss_SHA_P_train, loss_SHA_P_test, model_SHA_P 


@jit
def fit_model_SHA_T(X_train, y_train, X_test, y_test):
    # define model
    model_SHA_T = Sequential()
    
    model_SHA_T.add(LSTM(units = 8, return_sequences = True, input_shape = (X_train.shape[1], 8)))
    model_SHA_T.add(Dropout(0.2))
    
    model_SHA_T.add(LSTM(units = 8, return_sequences = True))
    model_SHA_T.add(Dropout(0.2))

    model_SHA_T.add(LSTM(units = 8))
    model_SHA_T.add(Dropout(0.2))
    
    
    model_SHA_T.add(Dense(units = 1))
    
    
    
    model_SHA_T.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    history = model_SHA_T.fit(X_train, y_train, epochs = 10, batch_size = 128, validation_data = (X_test, y_test),verbose=0)
    
    loss_SHA_T_train = history.history['loss'] #model_KWK_T.evaluate(X_test, y_test,verbose=0)
    
    loss_SHA_T_test = history.history['val_loss']
    
    return loss_SHA_T_train, loss_SHA_T_test, model_SHA_T 

@jit
def fit_model_KWK_T(X_train, y_train, X_test, y_test):
    # define model
    model_KWK_T = Sequential()
    
    model_KWK_T.add(LSTM(units = 10, return_sequences = True, input_shape = (X_train.shape[1], 10)))
    model_KWK_T.add(Dropout(0.2))


    model_KWK_T.add(LSTM(units = 10, return_sequences = True))
    model_KWK_T.add(Dropout(0.2))


    model_KWK_T.add(LSTM(units = 10))
    model_KWK_T.add(Dropout(0.2))
    
    model_KWK_T.add(Dense(units = 1))
    
    model_KWK_T.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    #model_KWK_T.fit(X_train, y_train, epochs = 100, batch_size = 128, verbose=0)
    
    history = model_KWK_T.fit(X_train, y_train, epochs = 10, batch_size = 128, validation_data = (X_test, y_test),verbose=0)
    
    loss_KWK_T_train = history.history['loss'] #model_KWK_T.evaluate(X_test, y_test,verbose=0)
    
    loss_KWK_T_test = history.history['val_loss']
    
    
    #loss_KWK_T_test = model_KWK_T.evaluate(X_test, y_test,verbose=0)
    
    #historty_KWK_T = model_KWK_T.fit(X_train, y_train, epochs = 100, batch_size = 128, verbose=0)
    
    #model_KWK_T.fit(X_train, y_train, epochs = 100, batch_size = 128, validation_data = (X_test, y_test),verbose=0)
    
    #loss_KWK_T_train = historty.historty['loss'] #model_KWK_T.evaluate(X_test, y_test,verbose=0)
    
    #loss_KWK_T_test = historty.historty['val_loss']
    
    return loss_KWK_T_train, loss_KWK_T_test, model_KWK_T

###############################################################################

loss_SHA_P_train, loss_SHA_P_test, model_SHA_P = fit_model_SHA_P(X_train_SHA_P, y_train_SHA_P,X_test_SHA_P, y_test_SHA_P)

loss_SHA_T_train, loss_SHA_T_test, model_SHA_T = fit_model_SHA_T(X_train_SHA_T, y_train_SHA_T,X_test_SHA_T, y_test_SHA_T)

loss_KWK_T_train, loss_KWK_T_test, model_KWK_T = fit_model_KWK_T(X_train_KWK_T, y_train_KWK_T,X_test_KWK_T, y_test_KWK_T)




###############################################################################
### SAVE MODELS

model_SHA_P.save('shasta_profile.h5')

model_SHA_T.save('shasta_temp.h5')

model_KWK_T.save('keswick_temp.h5')


model_SHA_P.save_weights('shasta_profile_weights.h5')

model_SHA_T.save_weights('shasta_temp_weights.h5')

model_KWK_T.save_weights('keswick_temp_weights.h5')




#loss_SHA_P_test = fit_model_SHA_P(X_train_SHA_P, y_train_SHA_P,X_test_SHA_P, y_test_SHA_P)
#
#loss_SHA_T_test = fit_model_SHA_T(X_train_SHA_T, y_train_SHA_T,X_test_SHA_T, y_test_SHA_T)
#
#loss_KWK_T_test = fit_model_KWK_T(X_train_KWK_T, y_train_KWK_T,X_test_KWK_T, y_test_KWK_T)




total_train_loss = loss_SHA_P_train + loss_SHA_T_train + loss_KWK_T_train 

total_test_loss = loss_SHA_P_test + loss_SHA_T_test + loss_KWK_T_test



from matplotlib import pyplot
pyplot.subplot(311)
pyplot.plot(loss_SHA_P_train, label='train')
pyplot.plot(loss_SHA_P_test, label='test')
pyplot.title('Shasta Profile Temp')
pyplot.legend()
pyplot.show()

pyplot.subplot(312)
pyplot.plot(loss_SHA_T_train, label='train')
pyplot.plot(loss_SHA_T_test, label='test')
pyplot.title('Shasta Outflow Temp')
pyplot.legend()
pyplot.show()


pyplot.subplot(313)
pyplot.plot(loss_KWK_T_train, label='train')
pyplot.plot(loss_KWK_T_test, label='test')
pyplot.title('Kesweick Outflow Temp')
pyplot.legend()
pyplot.show()



X_valdidation_SHA_P = []

X_valdidation_SHA_T = []

X_valdidation_KWK_T = []

for i in range(lag_day, len(validation_set_scaled_SHA_T)):
    
    X_valdidation_SHA_P.append(validation_set_scaled_SHA_P[i-lag_day:i, :]) #predictors of the year 2017
    
    X_valdidation_SHA_T.append(validation_set_scaled_SHA_T[i-lag_day:i, :])
    
    X_valdidation_KWK_T.append(validation_set_scaled_KWK_T[i-lag_day:i, :])


X_valdidation_SHA_P = np.array(X_valdidation_SHA_P)

X_valdidation_SHA_T = np.array(X_valdidation_SHA_T)

X_valdidation_KWK_T = np.array(X_valdidation_KWK_T)


# Reshaping
X_valdidation_SHA_P = np.reshape(X_valdidation_SHA_P, (X_valdidation_SHA_P.shape[0], X_valdidation_SHA_P.shape[1], 23))


X_valdidation_SHA_T = np.reshape(X_valdidation_SHA_T, (X_valdidation_SHA_T.shape[0], X_valdidation_SHA_T.shape[1], 8))


X_valdidation_KWK_T = np.reshape(X_valdidation_KWK_T, (X_valdidation_KWK_T.shape[0], X_valdidation_KWK_T.shape[1], 10))



predicted_scled_SHA_P = model_SHA_P.predict(X_valdidation_SHA_P)


predicted_scled_SHA_T = model_SHA_T.predict(X_valdidation_SHA_T)


predicted_scled_KWK_T = model_KWK_T.predict(X_valdidation_KWK_T)



sc1 = MinMaxScaler(feature_range = (0, 1)) 

sc2 = MinMaxScaler(feature_range = (0, 1))

sc3 = MinMaxScaler(feature_range = (0, 1)) 


data_all_SHA_P = sc1.fit_transform(dataset_tran_test_val_in[:,[26,27,28]])


data_all_SHA_T = sc2.fit_transform(dataset_tran_test_val_in[:,29].reshape(-1, 1))


data_all_KWK_T = sc3.fit_transform(dataset_tran_test_val_in[:,30].reshape(-1, 1))



data_observed_SHA_P = dataset_tran_test_val_in[6208:,[26, 27,28]]


data_observed_SHA_T = dataset_tran_test_val_in[6208:,29]


data_observed_KWK_T = dataset_tran_test_val_in[6208:,30]


#data_observed_middle = data_observed[:,0]


predicted_temp_SHA_P = sc1.inverse_transform(predicted_scled_SHA_P)


predicted_temp_SHA_T = sc2.inverse_transform(predicted_scled_SHA_T)


predicted_temp_KWK_T = sc3.inverse_transform(predicted_scled_KWK_T)


#############################################
#Result visualization

plt.plot(data_observed_SHA_P[lag_day:,0], color = 'red', label = 'Actual_Middle ')
plt.plot(predicted_temp_SHA_P[:,0], color = 'blue', label = 'Predicted_Middle')
plt.title('Shasta Profile Temprature Lower: 2017')
plt.ylim(7, 16)
plt.grid(True)
plt.xlabel('Time, # days')
plt.ylabel('Temperature, in oC')
plt.legend()
plt.show()




plt.plot(data_observed_SHA_P[lag_day:,1], color = 'red', label = 'Actual_Lower ')
plt.plot(predicted_temp_SHA_P[:,1], color = 'blue', label = 'Predicted_Lower')
plt.title('Shasta Profile Temprature Lower: 2017')
plt.ylim(7, 16)
plt.grid(True)
plt.xlabel('Time, # days')
plt.ylabel('Temperature, in oC')
plt.legend()
plt.show()


# Visualising the results: Raw
#plt.subplot(133)
plt.plot(data_observed_SHA_P[lag_day:,2], color = 'red', label = 'Actual_Side')
plt.plot(predicted_temp_SHA_P[:,2], color = 'blue', label = 'Predicted_Side')
plt.title('Shasta Profile Temprature Side: 2017')
plt.ylim(7, 16)
plt.grid(True)
plt.xlabel('Time, # days')
plt.ylabel('Temperature, in oC')
plt.legend()
plt.show()



plt.plot(data_observed_SHA_T[lag_day:], color = 'red', label = 'Actual_outflow temp ')
plt.plot(predicted_temp_SHA_T[:], color = 'blue', label = 'Predicted_outflow temp')
plt.title('Shasta outflow Temprature: 2017')
plt.ylim(7, 16)
plt.grid(True)
plt.xlabel('Time, # days')
plt.ylabel('Temperature, in oC')
plt.legend()
plt.show()




plt.plot(data_observed_KWK_T[lag_day:], color = 'red', label = 'Actual_Keswick outflow temp ')
plt.plot(predicted_temp_KWK_T[:], color = 'blue', label = 'Predicted_Keswick outflow temp')
plt.title('Keswick outflow Temprature: 2017')
plt.ylim(7, 16)
plt.grid(True)
plt.xlabel('Time, # days')
plt.ylabel('Temperature, in oC')
plt.legend()
plt.show()








from math import sqrt
from sklearn.metrics import mean_squared_error
rmse_SHA_P_middle = sqrt(mean_squared_error(data_observed_SHA_P[lag_day:,0], predicted_temp_SHA_P[:,0]))
rmse_SHA_P_low = sqrt(mean_squared_error(data_observed_SHA_P[lag_day:,1], predicted_temp_SHA_P[:,1]))
rmse_SHA_P_side = sqrt(mean_squared_error(data_observed_SHA_P[lag_day:,2], predicted_temp_SHA_P[:,2]))

rmse_SHA_T = sqrt(mean_squared_error(data_observed_SHA_T[lag_day:], predicted_temp_SHA_T[:]))
rmse_KWK_T = sqrt(mean_squared_error(data_observed_KWK_T[lag_day:], predicted_temp_KWK_T[:]))


print('Validation RMSE Shasta Profile Middle: %.3f' % rmse_SHA_P_middle)
print('Validation RMSE Shasta Profile Lower: %.3f' % rmse_SHA_P_low)
print('Validation RMSE Shasta Profile side: %.3f' % rmse_SHA_P_side)

print('Validation RMSE Shasta outflow temp: %.3f' % rmse_SHA_T)
print('Validation RMSE Keswick outflow temp: %.3f' % rmse_SHA_T)


stop = timeit.default_timer()

total_hr = (stop-start)/3600

print(stop-start)

print(total_hr)




