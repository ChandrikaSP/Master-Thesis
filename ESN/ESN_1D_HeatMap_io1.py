import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import warnings
import time
warnings.filterwarnings('ignore')
from pyESN import ESN
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.preprocessing import MinMaxScaler

##### split sequences into samples
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

data_tr_big = open("OnlyDY_2times.txt").read().split() #"amazon.txt" OnlyDY
data_tr_big = np.array(data_tr_big).astype('float64')
data_tr_1_1 = open("Only_DY1_1.txt").read().split() #"amazon.txt" OnlyDY
data_tr_1_1 = np.array(data_tr_1_1).astype('float64')
data_tr_1_2 = open("Only_DY2_1.txt").read().split() #"amazon.txt" OnlyDY
data_tr_1_2 = np.array(data_tr_1_2).astype('float64')
data_tr_1_3 = open("Only_DY1_3.txt").read().split() #"amazon.txt" OnlyDY
data_tr_1_3 = np.array(data_tr_1_3).astype('float64')
data_tr_2_1 = open("Only_DY2_1.txt").read().split() #"amazon.txt" OnlyDY
data_tr_2_1 = np.array(data_tr_2_1).astype('float64')
data_tr_2_2 = open("Only_DY2_2.txt").read().split() #"amazon.txt" OnlyDY
data_tr_2_2 = np.array(data_tr_2_2).astype('float64')
data_tr_2_3 = open("Only_DY2_3.txt").read().split() #"amazon.txt" OnlyDY
data_tr_2_3 = np.array(data_tr_2_3).astype('float64')
data_tr_3_1 = open("Only_DY3_1.txt").read().split() #"amazon.txt" OnlyDY
data_tr_3_1 = np.array(data_tr_3_1).astype('float64')
data_tr_3_2 = open("Only_DY3_2.txt").read().split() #"amazon.txt" OnlyDY
data_tr_3_2 = np.array(data_tr_3_2).astype('float64')
data_tr_3_3 = open("Only_DY3_3.txt").read().split() #"amazon.txt" OnlyDY
data_tr_3_3 = np.array(data_tr_3_3).astype('float64')
data_tr_5_1 = open("Only_DY5_1.txt").read().split() #"amazon.txt" OnlyDY
data_tr_5_1 = np.array(data_tr_5_1).astype('float64')
data_tr_5_2 = open("Only_DY5_2.txt").read().split() #"amazon.txt" OnlyDY
data_tr_5_2 = np.array(data_tr_5_2).astype('float64')
data_tr_5_3 = open("Only_DY5_3.txt").read().split() #"amazon.txt" OnlyDY
data_tr_5_3 = np.array(data_tr_5_3).astype('float64')
# data_tr_6_1 = open("Only_DY6_1.csv").read().split() #"amazon.txt" OnlyDY
# data_tr_6_1 = np.array(data_tr_6_1).astype('float64')
# data_tr_6_2 = open("Only_DY6_2.txt").read().split() #"amazon.txt" OnlyDY
# data_tr_6_2 = np.array(data_tr_6_2).astype('float64')
# data_tr_6_3 = open("Only_DY6_3.txt").read().split() #"amazon.txt" OnlyDY
# data_tr_6_3 = np.array(data_tr_6_3).astype('float64')

# data_t = open("Only_DY2.txt").read().split() #Only_DY2.txt  Versuch 6.1.csv
# data_t = np.array(data_t).astype('float64')

future = 10
in_1_1 = data_tr_1_1[:-future] #deleting last future number of values
out_1_1 = data_tr_1_1[future:] #deleting first future number of values
in_1_2 = data_tr_1_2[:-future] #deleting last future number of values
out_1_2 = data_tr_1_2[future:] #deleting first future number of values
in_1_3 = data_tr_1_3[:-future] #deleting last future number of values
out_1_3 = data_tr_1_3[future:] #deleting first future number of values
in_2_1 = data_tr_2_1[:-future] #deleting last future number of values
out_2_1 = data_tr_2_1[future:] #deleting first future number of values

n_past = 20
n_future = 5

data_raw=data_tr_1_1.reshape(920,1)
scaler = MinMaxScaler()
scaler.fit(data_raw.reshape(-1,1))

GPR=np.load('920_75_realz.npy')
GPR=GPR.T
GPR_GT = np.load('xyEXACT_2.npy')
GPR_GT = scaler.inverse_transform(GPR_GT)
GPR_GT=(GPR_GT[:,1]).reshape(-1,1)  #to get (920,2) 

GT_synthetic = open("OnlyDY_GT_synthetic.csv").read().split()
GT_synthetic = np.array(GT_synthetic).astype('float64')
#creating one array GPR 25samples data: 23000 for input
one_array_GPR=[]
for i in range(75):  #25
    one_array_GPR=np.hstack((one_array_GPR,GPR[i]))
print(np.shape(one_array_GPR))

GPR_in, GPR_out = [],[]
# GPR_in, GPR_out = split_sequences(one_array_GPR, n_past, n_future, 1)
GPR_in, GPR_out = split_sequences_new(one_array_GPR, n_past, n_future, 1)

GPR_in=np.array(GPR_in)
GPR_out=np.array(GPR_out)
print("GPR SS func. i/p and o/p shape:", np.shape(GPR_in), np.shape(GPR_out))

##creating output(23000,)->(22975,20)(22975,5) for training using original, exact / GT data as input
GT_25samples=[]
for i in range(75):  #25
    GT_25samples=np.hstack((GT_25samples,GT_synthetic))
print(np.shape(GT_25samples))

GT_in, GT_out = [],[]
# GT_in, GT_out = split_sequences(GT_25samples, n_past, n_future, 1)
GT_in, GT_out = split_sequences_new(GT_25samples, n_past, n_future, 1)

GT_in=np.array(GT_in)
GT_out=np.array(GT_out)
print("GT SS func. i/p & o/p shape:", np.shape(GT_in), np.shape(GT_out))

in_data_raw, out_data_raw = split_sequences_new(data_tr_1_1, n_past, n_future, 1)
in_data_train, out_data_train = split_sequences_new(data_tr_1_3, n_past, n_future, 1)

rand_seed=23

# store best parameter set #
n_reservoir = [50,70,80,90,100,110,150,120,150,170,200] #[ 3,6,7,8,9,10,12,13,19,24,36,50,70]  #10 values
spectral_radius_set = [0.5,0.6,0.7,0.8,0.9,0.99,1,1.1,1.2,1.4,1.5,1.7,1.8,1.9,2] #[0.5, 0.7, 0.9,  1,  1.1,1.2,1.7,2,3,5]  #14 values 
# sparsity_set = [0.2,0.3,0.4,0.5,0.6,0.7,0.9]  #not more than 1 bcz of SVD didn't converge error
# sparsity_set = [ 0.0001, 0.0003,0.0007, 0.001, 0.003, 0.005, 0.007,0.01]  #noise

n_reservoir_set_size=len(n_reservoir)  #n_reservoir  spectral_radius_set
radius_set_size  = len(spectral_radius_set)  #sparsity_set
loss_l=np.zeros([n_reservoir_set_size, radius_set_size])
loss = np.zeros([n_reservoir_set_size, radius_set_size])
aa,bb,cc=[],[],[]
for l in range(n_reservoir_set_size):
    n_reservoir_ = n_reservoir[l]  #n_reservoir  spectral_radius_set

    for j in range(radius_set_size):
        noise = .0005  #sparsity_set
        esn = ESN(n_inputs = 20, #1, 20  1
        n_outputs = 5, #1, 5  1
        n_reservoir = n_reservoir[l], #n_reservoir[l] 12
        sparsity=0.3, #sparsity_set 0.3 ###################### CHANGE SPARSITY VALUE BETWEEN 0.2,0.3 and 0.4  ###########
        random_state=rand_seed,
        spectral_radius = spectral_radius_set[j], #radius_set[j]  sparsity_set
        noise=.0005,
        teacher_forcing=False) #.0005 False

        # pred_training = esn.fit(in_, out_)
        # pred_tot = esn.predict(t_in_, continuation=True)

        pred_ =[]
        n=100                          ####IMP CODE###########
        m=int((len(in_1_1))/20)
        print(m)
        col = ['k', 'g', 'r', 'b', 'm', 'c','y']
        for i in range(4):
            test=data_tr_1_3[n:n+20]
            # pred_train=esn.fit(GPR_in, GT_out)  #in_1_1, out_1_1
            # pred_train=esn.fit(in_1_3, out_1_3)
            pred_train=esn.fit(GPR_in, GT_out)  #in_1_2, out_1_2  GPR_in, GT_out  in_data_train, out_data_train
            current_pred=esn.predict(in_data_raw,continuation=False)       
            pred_.append(current_pred)
            n=n+20

            loss[l, j] = (mean_squared_error(out_data_raw, current_pred)) #MSE(pred_tot, t_out_) data_tr_1_3[n+10:n+20+10]    np.sqrt
            # loss_l[l,j]=loss[l,j]       
        # print('n_reservoir_ = ', n_reservoir[l], ', radius = ', spectral_radius_set[j], ', MSE = ', loss[l][j] )
        aa.append(loss[l][j])
        bb.append(n_reservoir[l]) #n_reservoir  spectral_radius_set
        cc.append(spectral_radius_set[j])  #sparsity_set
    # time.sleep(2)

res1 = []
for x,y,z in zip(aa, bb,cc):
    res1.append((x,y,z))
res1_t=np.transpose(res1)
res1_t=res1_t.tolist()
dd=res1_t[0].index(np.min(res1_t[0]))
print('min loss val:',dd)
print('min loss parameter set:',res1[dd])

plt.figure(figsize=(16,8))
im = plt.imshow(loss.T, vmin=abs(loss).min(), vmax=abs(loss).max(), origin='lower',cmap='PuRd')
plt.xticks(np.linspace(0,n_reservoir_set_size-1,n_reservoir_set_size), n_reservoir)  #n_reservoir  spectral_radius_set
plt.yticks(np.linspace(0,radius_set_size-1, radius_set_size), spectral_radius_set)  #sparsity_set
plt.xlabel('n_reservoirs', fontsize=16) #n_reservoirs
plt.ylabel('spectral radius', fontsize=16)  #spectral_radius  

# im.set_interpolation('bilinear')
cb = plt.colorbar(im)

plt.legend()
plt.grid()
plt.show()
