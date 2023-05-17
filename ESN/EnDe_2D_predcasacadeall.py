# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import statsmodels.api as sm
# from keras.layers import LSTM, Dense, Bidirectional, TimeDistributed, Conv1D, MaxPooling1D, Flatten, ConvLSTM2D, InputLayer, Input, Reshape, Conv2D, Activation
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.optimizers import Adam
# import time
# from sklearn.preprocessing import MinMaxScaler
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


def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]  #end_ix
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

FF1_1=pd.read_csv('dxy1.csv')
FF1_2=pd.read_csv('dxy1_2.csv')
FF1_3=pd.read_csv('dxy1_3.csv')   

# FF1_2_test=FF1_2[360:390]  #[360:390] #440:470  280:300 320:340 360:390  720:750

FF1_1_np=FF1_1.to_numpy()
# FF1_1_list=[]
# FF1_1_list=FF1_1_np.tolist()
FF1_2_np=FF1_2.to_numpy()
# FF1_2_list=[]
# FF1_2_list=FF1_2_np.tolist()
FF1_3_np=FF1_3.to_numpy()
# FF1_3_list=[]
# FF1_3_list=FF1_3_np.tolist()
n_steps_in, n_steps_out = 9, 20  #20 5  HERE WE SET NO. OF PAST I/P & FUTURE O/P VALUES AS NEEDED
n=180

scaler = MinMaxScaler()
scaler.fit(FF1_1)

scaled_train1 = scaler.transform(FF1_1)
scaled_train2 = scaler.transform(FF1_2)
# scaled_train3 = scaler.transform(FF1_3) #look into it

# X_train1, X_test1, y_train1, y_test1 = train_test_split( scaled_train1[:,0],  scaled_train1[:,1], test_size=0.1,shuffle=False)
# print(X_train1.shape, X_test1.shape, y_train1.shape, y_test1.shape)
# X_train2, X_test2, y_train2, y_test2 = train_test_split( scaled_train2[:,0],  scaled_train2[:,1], test_size=0.1,shuffle=False)
# X_train3, X_test3, y_train3, y_test3 = train_test_split( scaled_train3[:,0],  scaled_train3[:,1], test_size=0.1,shuffle=False)

# datatrain1= np.column_stack((X_train1, y_train1))
# datatrain2= np.column_stack((X_train2, y_train2))
# # datatrain3= np.column_stack((X_train3, y_train3))
# datatest1= np.column_stack((X_test1, y_test1))
# datatest2= np.column_stack((X_test2, y_test2))
# datatest3= np.column_stack((X_test3, y_test3))

# n_steps = 9 #10 time steps i/p are used and 1 time step prediction is the output
data_train1, data_train_label1 = split_sequences(scaled_train1, n_steps_in, n_steps_out) #(FF1_1_list, n_steps) data_train is now in the format of [samples, timesteps, features]
data_train2, data_train_label2 = split_sequences(scaled_train2, n_steps_in, n_steps_out) #(FF1_2_list, n_steps)
# data_train3, data_train_label3 = split_sequences(datatrain3, n_steps_in, n_steps_out)

# data_test1, data_test_label1 = split_sequences(datatest1, n_steps_in, n_steps_out) #(FF1_1_list, n_steps) data_train is now in the format of [samples, timesteps, features]
# data_test2, data_test_label2 = split_sequences(datatest2, n_steps_in, n_steps_out) #(FF1_2_list, n_steps)
# data_test3, data_test_label3 = split_sequences(datatest3, n_steps_in, n_steps_out)
n_features = data_train1.shape[2]
# print('data_test1',data_test1)  #its in 0& 1 format ie., data is normalized so cant see the actual values here
# print('data_test1_y',data_test_label1)

# X, y=split_sequences(FF1_1_np, n_steps_in, n_steps_out)
# X_test, y_test=split_sequences(FF1_2_np, n_steps_in, n_steps_out)
# X = tf.convert_to_tensor(X, dtype=tf.float32) # y = tf.convert_to_tensor(y, dtype=tf.float32)
# n_features=X.shape[2]
print(data_train1.shape, data_train_label1.shape)  #X.shape, y.shape
# print('n_fea',n_features)
# # summarize the data
# for i in range(len(X)):
# 	print(X[i], y[i])

model=Sequential()
activation=LeakyReLU(alpha=0.2)   
model.add(LSTM(24,activation=activation, input_shape=(n_steps_in, n_features))) #, return_sequences=True 200 32  'relu' 
model.add(RepeatVector(n_steps_out))
model.add(LSTM(24,activation=activation, return_sequences=True)) # , input_shape=(n_steps,n_features)  200  24 'relu'
model.add(TimeDistributed(Dense(n_features)))  #added newly 25.02 "where each neuron receives input from all the neurons of previous layer, thus called as dense"

model.compile(optimizer='adam', loss='mse')
history=model.fit(data_train1, data_train_label1, epochs=200, batch_size=256, verbose=2, shuffle=False) #, batch_size=72 batch_size=256,validation_data=(data_train1, data_train_label1),
time.sleep(2)
history=model.fit(data_train2, data_train_label2, epochs=200, batch_size=256, verbose=2, shuffle=False)
# model.save('EnDeLSTM_2D_nonARMA_24nodes_200e__n_future:5final.h5') #EnDeLSTM_2D_nonARMA_24nodes_200e__n_future:5final.h5

# model = load_model('EnDeLSTM_2D_nonARMA_24nodes_100e_final.h5') #EnDe2D_PredCascade.h5  EnDeLSTM_2D_nonARMA_24nodes_100e_final.h5
# history=model.fit(data_train3, data_train_label3, epochs=500, verbose=2, shuffle=False)
# model.fit(x=data_train1, y=data_train_label1, epochs=2, verbose=2, shuffle=False) #here now batch_size=256,
# time.sleep(3)
# model.fit(x=data_train2, y=data_train_label2, epochs=2, verbose=2, shuffle=False)  # batch_size=256,
# model.summary()
# x_input = np.array([[60, 65, 125], [70, 75, 145], [80, 85, 165]])
# x_input = x_input.reshape((1, n_steps_in, n_features))
# yhat = model.predict(x_input, verbose=0)
# print(yhat)
# predictions1=[]
# first_batch1=data_test1[-1:] #[-6:]  0:n_steps  X_test
# first_batch_y1=data_test_label1[-1:]  #y_test
# current_batch1=first_batch1.reshape((1,n_steps_in,n_features))
# print('first_batch',first_batch)
# print('current_batch_input',current_batch)

# for i in range(len(X_test)):
# 	current_pred=model.predict(current_batch)[0]
# 	#print('current_batch' ,current_batch)
# 	#append predictions into array
# 	predictions.append(current_pred)
# 	#remove the first value
# 	current_batch_rmv_first=current_batch[:,1:,:]
# 	#update the batch(with first current_batch_rmv_first values then prediction in current_pred values)
# 	current_batch=np.append(current_batch_rmv_first,[[current_pred]],axis=1)

# predictions1 = model.predict(current_batch1)[0]
# # print('current_batch1' ,current_batch)
# true_predictions1 = scaler.inverse_transform(predictions1)
# first_batch1 = scaler.inverse_transform(first_batch1[0])
# first_batch_y1 = scaler.inverse_transform(first_batch_y1[0])
# print('testing_input_past:',first_batch1)
# print('testing_GT_future_pred_values',first_batch_y1)
# print('predictions_1:',true_predictions1)

no_pred_tohappem=int(len(FF1_3)/5) #20
all_pred=[]
import time

df_true_predictions = pd.DataFrame(columns=['pdx','pdy'])
for i in range(no_pred_tohappem):
    print('nnnnnnn at 1',n)
    test = FF1_3.iloc[n:n+n_steps_in]  #20
    # print(test)
    scaled_test = scaler.fit_transform(test)
    test_predictions = []
    # current_batch=np.empty((1,4,1))
    # first_eval_batch=np.empty((4,1))

    first_eval_batch = scaled_test[0:n_steps_in]  #scaled_train             
    current_batch = first_eval_batch.reshape((1, n_steps_in, n_features))
    for i in range(len(scaled_test)):
        current_pred=model.predict(current_batch)[0]
        # print('current_batch' ,current_batch)
        # print('rr',current_batch[0])
        #append predictions into array   
        test_predictions.append(current_pred) 
        # current_pred=current_pred.reshape(1,4,1)
        #remove the first value
        # current_batch_rmv_first=current_batch[:,1:,:]       
        #update the batch(with first current_batch_rmv_first values then prediction in current_pred values)
        # current_batch=np.append(current_batch_rmv_first,[[current_pred]],axis=1)
        # all_pred=np.append(current_batch_rmv_first[0],current_pred)

    true_predictions = scaler.inverse_transform(test_predictions[0])
    # time.sleep(3)
    # calculate RMSE
    # rmse = np.sqrt(mean_squared_error(first_batch_y, true_predictions))
    # print('Test RMSE: %.3f' % rmse)

    df_true_predictions = df_true_predictions.append(pd.DataFrame(true_predictions))
    
    # test['Predictions'] = true_predictions
    # print("test with Prediction values:",test)

    # print(df_OldTest)
    # df_OldTest = df_OldTest.append(pd.DataFrame(test))
    # # df = df.append(pd.DataFrame( list,columns=[ 'Name', 'Age', 'City', 'Country']), ignore_index = True)
    # print("added DF",df_OldTest)
    n=n+4 #20 number of row indices by which pred will be moved by for next pred to happen
    print('nnnnnnnnnnnnnnnnnnn at 22222',n)
    if n>1300:  #919  1300
        break
df_true_predictions.columns = ['pdx', 'pdy','nan', 'nan1']
df_true_predictions.to_csv('EnDe_PredCascade_3.csv')
# test.plot(figsize=(14,5))
# df1_1.plot(figsize=(14,5))
# df1_1.plot(figsize=(14,5),color='b')
# df1_2.plot(figsize=(14,5)) #plot of time vs dx or dy value
plt.scatter(FF1_3['dx'],FF1_3['dy'],c='y',label="Ground-Truth data")
plt.scatter(df_true_predictions['nan'],df_true_predictions['nan1'],c='olive',label="Predictions")
plt.xlabel('Flap displacement dX values in ms')
plt.ylabel('Flap displacement dY values in mm')
plt.title('Encoder-decoder LSTM time-series prediction (multi-step: 20 future values)')
plt.legend(loc='upper right')
plt.grid()
# plt.plot(df1_1['dx'],df1_1['dy'],color='y')
# # plt.plot(df1_2['dx'],df1_2['dy'],color='b')
# test['Predictions'].plot(figsize=(14,5))
plt.show()

