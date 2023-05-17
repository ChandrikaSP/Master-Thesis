from numpy import array
from keras.layers import LSTM
from keras.layers import Dense
import os
import glob
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, TimeDistributed, Conv1D, MaxPooling1D, Flatten, ConvLSTM2D, InputLayer, Input, Reshape, Conv2D, Activation
from numpy import hstack
from keras.models import Model
from time import time
from keras.preprocessing import image as im
import pickle
import time

from copy import deepcopy
from sklearn.model_selection import train_test_split
from keras import backend as K
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# split a univariate sequence into samples
def split_sequences(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
# os.getcwd()
# os.chdir('Tipping-Tank-csv/Versuch_all')
FF1_1=pd.read_csv('dxy1.csv')
FF1_2=pd.read_csv('dxy1_2.csv')
FF1_3=pd.read_csv('dxy1_3.csv')

FF1_2_test=FF1_2[360:390] #440:470  280:300 320:340 360:390  720:750

FF1_1_np=FF1_1.to_numpy()
FF1_1_list=[]
FF1_1_list=FF1_1_np.tolist()
FF1_2_np=FF1_2_test.to_numpy()
FF1_2_list=[]
FF1_2_list=FF1_2_np.tolist()
FF1_3_np=FF1_3.to_numpy()
FF1_3_list=[]
FF1_3_list=FF1_3_np.tolist()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(FF1_1)

scaled_train1 = scaler.transform(FF1_1)
scaled_train2 = scaler.transform(FF1_3)
scaled_test = scaler.transform(FF1_2_test) #look into it
n_steps = 9 #10 time steps i/p are used and 1 time step prediction is the output
data_train1, data_train_label1 = split_sequences(scaled_train1, n_steps) #(FF1_1_list, n_steps) data_train is now in the format of [samples, timesteps, features]
data_train2, data_train_label2 = split_sequences(scaled_train2, n_steps) #(FF1_2_list, n_steps)
data_test, data_test_label = split_sequences(scaled_test, n_steps)
n_features = data_train1.shape[2]
# from keras.preprocessing.sequence import TimeseriesGenerator
# generator1 = TimeseriesGenerator(scaled_train1, scaled_train1, length=n_steps, batch_size=1)
# generator2 = TimeseriesGenerator(scaled_train2, scaled_train2, length=n_steps, batch_size=1)
# X,y = generator1[0]
# print(f'Given the Array: \n{X.flatten()}')
# print(f'Predict this y: \n {y}')
print(np.shape(data_train1))
print(np.shape(data_train_label1))
# print('len(Gen)',len(generator1))
print(data_train_label1)

# print(np.shape(generator2[1]))
# g_x,g_y=generator

# new_data_train = tf.convert_to_tensor(data_train, dtype=tf.float32)
# new_data_train_label = tf.convert_to_tensor(data_train_label)
# new_data_val = tf.convert_to_tensor(data_val, dtype=tf.float32)
# new_data_val_label = tf.convert_to_tensor(data_val_label)
# X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
# y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

# # define model
model = Sequential()
model.add(LSTM(32, activation='relu', input_shape=(n_steps, n_features)))  #one hidden LSTM layer 
model.add(Dense(n_features))                                              #n_features number of/=2 Dense output layers
model.add(Activation("linear"))
# model = Sequential()
# model.add(LSTM(32, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))  #one hidden LSTM layer #before 200
# model.add(LSTM(30, activation='relu')) #remove it  before 100
# # model.add(Dense(30, activation='relu'))  #added newly 25.02
# model.add(Dense(n_features))                                              #n_features number of/=2 Dense output layers
# model.add(Activation("linear"))

model.compile(optimizer='adam', loss='mse')
# model.summary()
# # fit model
history=model.fit(x=data_train1, y=data_train_label1, epochs=600, batch_size=256,verbose=2, shuffle=False) #here now batch_size=256,
time.sleep(3)
history=model.fit(x=data_train2, y=data_train_label2, epochs=600, batch_size=256,verbose=2, shuffle=False)  # batch_size=256,
# loss_per_epoch = model.history.history['loss']
# plt.plot(range(len(loss_per_epoch)),loss_per_epoch)

# history1 = model.fit(generator1, epochs=20, verbose=1)
# history2 = model.fit(generator2, epochs=20, verbose=1)
# model.save('vanillaLSTM_32nodes_600e_2d_i9o1_final.h5')
# model = load_model('vanillaLSTM_32nodes_600e_2d_i9o1_final.h5') #retraining the model  vanillaLSTM_32nodes_approach2_2D_batchsz256_epoch100.h5
model.summary()
print('yes done************')
print(n_features)
predictions=[]

# first_batch=scaled_test[0:15] #[-6:]
first_batch=data_test[-1:] #[-6:]  0:n_steps
current_batch=first_batch.reshape((1,n_steps,n_features))
# print('first_batch',first_batch)
# print('current_batch_input',current_batch)

for i in range(len(data_test)):
	current_pred=model.predict(current_batch)[0]
	#print('current_batch' ,current_batch)
	#append predictions into array
	predictions.append(current_pred)
	#remove the first value
	current_batch_rmv_first=current_batch[:,1:,:]
	#update the batch(with first current_batch_rmv_first values then prediction in current_pred values)
	current_batch=np.append(current_batch_rmv_first,[[current_pred]],axis=1)

# print('current_batch1' ,current_batch)
true_predictions = scaler.inverse_transform(predictions)
# true_predictions = predictions

# print(np.shape())
print('true_pred shape',np.shape(true_predictions))
# print(FF1_2_test)
df_true_predictions = pd.DataFrame(columns=['pdx','pdy'])
df_true_predictions = df_true_predictions.append(pd.DataFrame(true_predictions))
df_true_predictions.columns = ['nan', 'nan1', 'pdx', 'pdy']
df_true_predictions.to_csv('AllPred_val_VanillaLSTM_500e_2D.csv')

# scaled_test['Predictions'] = true_predictions
# print("test with Prediction values:",test)

plt.plot(FF1_2_test['dx'],FF1_2_test['dy'],color='g',label='test i/p data')
plt.plot(FF1_1['dx'],FF1_1['dy'],color='y', label='train data')
plt.plot(df_true_predictions['pdx'],df_true_predictions['pdy'], color='r', label='predictions')
plt.xlabel('Flap displacement (dX) values in ms')
plt.ylabel('Flap displacement (dY) values in mm')
plt.title("Vanilla LSTM 2D prediction plot")
plt.grid()
plt.legend()
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
plt.show()
#print('pred', predictions)