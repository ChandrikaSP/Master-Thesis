################################################## Final Stacked LSTM 2D both flap displ. + force TSF ##############################################################
from numpy import array
from keras.models import Sequential
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
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]  #tune it
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

os.getcwd()
# os.chdir('Tipping-Tank-csv/Versuch_all')

df=pd.read_csv('forceIntr_flap_1.1.csv')
#df=pd.read_csv('forceIntr_flap_1.2.csv')
#df=pd.read_csv('force&flapDisplacement_1.csv')
# data_raw = open("OnlyDY.txt").read().split() #"amazon.txt" OnlyDY
# data_raw = np.array(data_raw).astype('float64')

df_np=np.empty([587,5])
df_np=df.to_numpy()
df_list=[]
df_list=df_np.tolist()

train_data = df_list[:-50]
validation_data =df_list[-50:]

n_steps = 40 #10 time steps i/p are used and 1 time step prediction is the output
data_train, data_train_label = split_sequences(train_data, n_steps) # data_train is now in the format of [samples, timesteps, features]
data_val, data_val_label = split_sequences(validation_data, n_steps)
n_features = data_train.shape[2]

new_data_train = tf.convert_to_tensor(data_train, dtype=tf.float32)
new_data_train_label = tf.convert_to_tensor(data_train_label)
new_data_val = tf.convert_to_tensor(data_val, dtype=tf.float32)
new_data_val_label = tf.convert_to_tensor(data_val_label)

# X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
# y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

# # define model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))  #one hidden LSTM layer #before 200
model.add(LSTM(30, activation='relu')) #remove it  before 100
model.add(Dense(30, activation='relu'))  #added newly 25.02
model.add(Dense(n_features))                                              #n_features number of/=2 Dense output layers
model.add(Activation("linear"))  # this is optional

# Compile the model=Once the model is created, you can config the model with losses and metrics with model.compile()
#opt = keras.optimizers.Adam(learning_rate=0.01)
#model.compile(optimizer=opt, loss='mse')  #or loss='categorical_crossentropy'
model.compile(optimizer='adam', loss='mse')

# Give a summary
model.summary()

# fit model=training the model
history = model.fit(x=data_train, y=data_train_label, validation_data=(new_data_val, new_data_val_label), epochs=600, verbose=1)  #

# model.save('stackedLSTM_500e_50,30_final.h5')

# model = load_model('stackedLSTM_500e_50,30_final.h5')  #partly_trainedModel

print('yes done************')

predictions=[]

first_batch=data_val[-1:]
current_batch=first_batch.reshape((1,n_steps,n_features))
print('first_batch',first_batch)
print('current_batch_input',current_batch)

for i in range(len(data_val)):
	current_pred=model.predict(current_batch)[0]
	#print('current_batch' ,current_batch)
	#append predictions into array
	predictions.append(current_pred)
	#remove the first value
	current_batch_rmv_first=current_batch[:,1:,:]
	#update the batch(with first current_batch_rmv_first values then prediction in current_pred values)
	current_batch=np.append(current_batch_rmv_first,[[current_pred]],axis=1)

print('current_batch1' ,current_batch)

print(np.shape(first_batch))
print(type(current_batch))
print(np.shape(current_batch))
# plt.plot(current_batch[],current_batch[])
x_1=current_batch[0][:,0]
y_1=current_batch[0][:,1]
plt.plot(x_1, y_1,'b',label='flap displ. pred')
plt.plot(df['dx'],df['dy'],'r', label='flap displ. train i/p')

x_2=current_batch[0][:,2]
y_2=current_batch[0][:,3]
plt.plot(x_2, y_2,'k',label='force pred')
plt.plot(df['fx'],df['fy'],'y', label='force train i/p')
plt.xlabel('Flap displacement (dX) values in ms')
plt.ylabel('Flap displacement (dY) values in mm')
plt.title('Stacked LSTM 2D predictions')
plt.grid()
plt.legend()
# x_1=current_batch[0][:,0]
# y_1=current_batch[0][:,1]
# plt.plot(x_1, y_1,'b')
# plt.plot()
# plt.plot(df['force_dx'],df
# ['force_dY'],'r')

# x_2=current_batch[0][:,2]
# y_2=current_batch[0][:,3]
# plt.plot(x_2, y_2,'k')
# plt.plot(df['flap_dX'],df['flap_dY'],'y')

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
plt.show()
#print('pred', predictions)