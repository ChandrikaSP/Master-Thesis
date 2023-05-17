from dataclasses import dataclass
import os
import glob
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, TimeDistributed, Conv1D, MaxPooling1D, Flatten, ConvLSTM2D, InputLayer, Input, Reshape, Conv2D, Activation, RepeatVector, TimeDistributed
from keras.layers import LeakyReLU
from numpy import hstack
from keras.models import Model
from time import time
from keras.preprocessing import image as im
import pickle
from math import *
import time
from matplotlib import pyplot
from copy import deepcopy
from sklearn.model_selection import train_test_split
from keras import backend as K
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from pyESN import ESN 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# multivariate output data prep
from numpy import array
from numpy import hstack

# split a multivariate sequence into samples
def split_sequences(sequence, n_past_steps, n_future_steps, stride):
    """
    Divides a equidistant sequence into two sets. The first is a set of overlapping sub-sequences of
    n_past_steps entries starting at every stride's index. The second is a set of corresponding
    n_future_steps next values. n_features must be in axis 1.
    """
    X, Y = list(), list()
    for i in range(sequence.shape[0]):
        end_ix = i + n_past_steps + n_future_steps

        if i % stride != 0: continue
        if end_ix > sequence.shape[0] - 1: break

        seq_x, seq_y = sequence[i:i + n_past_steps], sequence[i + n_past_steps:end_ix]
        X.append(seq_x)
        Y.append(seq_y)
    return np.array(X), np.array(Y)

def split_sequences_new(sequence, n_past_steps, n_future_steps, stride):
    stride=1
    X, Y = list(), list()
    x=0
    for i in range(sequence.shape[0]):

        n_past_steps = 20
        n_future_steps = 5
        end_ix = x + n_past_steps + n_future_steps

        if i % stride != 0: continue
        if end_ix > sequence.shape[0] - 1: break

        seq_x, seq_y = sequence[x:x + n_past_steps], sequence[x + n_past_steps:end_ix]
        x = end_ix #0 25 50 75
    #     print(x)
        X.append(seq_x)
        Y.append(seq_y)
    return np.array(X), np.array(Y) 

def MSE(expected_out, actual_out):
    mse = (mean_squared_error(expected_out, actual_out))  ##(GT/expected_output,actual_output)  np.sqrt
    print('Test MSE: %.3f' % mse)

data_raw = open("Only_DY1_1.txt").read().split() #"amazon.txt" OnlyDY
data_raw = np.array(data_raw).astype('float64')
data_tr_1_2 = open("Only_DY1_2.txt").read().split() #"amazon.txt" OnlyDY
data_tr_1_2 = np.array(data_tr_1_2).astype('float64')
data_tr_1_3 = open("Only_DY1_3.txt").read().split() #"amazon.txt" OnlyDY
data_tr_1_3 = np.array(data_tr_1_3).astype('float64')

print(np.shape(data_raw))
data_raw=data_raw.reshape(920,1)
scaler = MinMaxScaler()
scaler.fit(data_raw.reshape(-1,1))

data_tr_1_1 = data_raw.reshape((920))
np.shape(data_tr_1_1)

n_past = 20
n_future = 5

# n_steps = 3#, n_features, 1
n_steps, n_features = 20, 1  #20  1
n_steps_out = 5  #5  1

in_data_raw, out_data_raw = split_sequences(data_raw, n_past, n_future, 1)
# in_data_raw, out_data_raw = split_sequences_new(data_raw, n_past, n_future, 1)
# in_data_test, out_data_test = split_sequences_new(data_tr_1_2, n_past, n_future, 1)

model=Sequential()  #clean code, task &comparison 
activation=LeakyReLU(alpha=0.2)   
model.add(LSTM(24,activation=activation, input_shape=(n_steps, n_features))) #, return_sequences=True 200 32  'relu'    24
model.add(RepeatVector(n_steps_out))
model.add(LSTM(24,activation=activation, return_sequences=True)) # , input_shape=(n_steps,n_features)  200  24 'relu'
model.add(TimeDistributed(Dense(n_features)))  #added newly 25.02 

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.summary()

### train with data_raw(920,1) using SplitSequeneces func, of (20,5). so i/p: (895,20) o/p: (895,5)
history=model.fit(in_data_raw, out_data_raw, validation_split=0.33, epochs=300, verbose=2, shuffle=False) #, batch_size=72 batch_size=256,validation_data=(data_train1, data_train_label1),

# model.save('EnDeLSTM_1D_nonGPR_200e_24nodes_i20o5_final.h5') #EnDeLSTM_1D_GPR_600e_75gpr_24nodes_final.h5
# model = load_model('EnDeLSTM_1D_nonGPR_200e_24nodes_i20o5_final.h5')  # EnDeLSTM_1D_GPR_75samples_200e.h5

n_idx, n_steps_in = 0, 20  #start index value  20
n_idx = pred_str_idx = n_idx
no_pred_tohappen=int(len(data_tr_1_2)/5) #20

df_pred = pd.DataFrame(columns=['pdy'])

for i in range(no_pred_tohappen):
    print('n[1] value:',n_idx)
    test = data_tr_1_2[n_idx:n_idx+n_steps_in]  #20   "HERE SPECIFY which TEST DATA to be used for EVALUATION / PREDICTION"
    # print(test)
    test_predictions = []
    
    first_eval_batch = test[0:n_steps_in]  # 20 values as input for prediction (20,1)       
    current_batch = first_eval_batch.reshape((1, n_steps_in, n_features)) #(1, 20, 1)
    loop=3  #in AR way, how many times do we want the model to perform FP
    for i in range((loop)):
        current_pred=model.predict(current_batch)[0]
        test_predictions.append(current_pred) 
        current_pred=current_pred.reshape(1,5,1)
        current_batch_rmv_first=current_batch[:,5:,:]
        #update the batch(with first current_batch_rmv_first values then prediction in current_pred values)
        current_batch=np.append(current_batch_rmv_first, current_pred,axis=1)  #[[current_pred[0]]]
    print('shape of test_pred', np.shape(test_predictions[0]))

    b=np.empty((1,1))  #creating 1D array having all pred values, for plotting and saving the values easily
    for i in range(loop):  
        b=np.append(b,test_predictions[i],axis=0)

    df_pred = df_pred.append(pd.DataFrame(b[1:]))  #test_predictions[0]
    # df_pred = df_pred.append(pd.DataFrame(test_predictions[0]))
    n_idx=n_idx+15 #20 number of row indices by which pred will be moved by for next pred to happen  4 20 6 15 (15final)  [Since loop=3, 15 values are predicted]
    print('n[2] value:',n_idx)
    time.sleep(1)
    if n_idx>895 :  #919  """n value where pred should stop, end index value  895"""
        break
df_pred.columns = [ 'pdy','nan']
df_pred.to_csv('EnDeLSTM_GPRPredCascade_4_data1.2_try.csv')  #EnDeLSTM_GPRPredCascade_4_data1.2_try

df=pd.read_csv('EnDeLSTM_GPRPredCascade_4_data1.2_try.csv')
print(np.shape(df))
shape=(np.shape(df)[0])
plt.scatter(np.arange(0,np.shape(data_tr_1_2)[0]),data_tr_1_2,c='y',label="Ground-Truth data")
plt.scatter(np.arange(20,shape+20),df_pred['nan'],c='olive',label="Predictions")
plt.xlabel('Time index (in millisecond [ms])')
plt.ylabel('Flap displacement dY value (in millimeter [mm])')
plt.grid()

MSE(data_tr_1_2[pred_str_idx:n_idx], df['nan'])

plt.legend()
plt.show()
# plot history
loss=history.history['loss']
accuracy=history.history['accuracy']
loss_val = history.history['val_accuracy']

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# summarize history for loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.legend()
plt.show()