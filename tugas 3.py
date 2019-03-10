# -*- coding: utf-8 -*-

import numpy as py
import pandas as pd
import math
import matplotlib.pyplot as plt

idx = ['x1','x2','x3','x4','types']
df = pd.read_csv('iris.csv',names=idx)

df.head(101)

data = df.head(101).values.tolist()

for i in data:
	if(i[4]=='Setosa'):
		i.append(0)
	else:
		i.append(1)

data[0][5]



array1 = data[1:20] 
array2 = data[21:40]
array3 = data[41:60]
array4 = data[61:80]
array5 = data[81:100] 

train1 = array2[:] + array3[:] + array4[:] + array5[:]
train2 = array1[:] + array3[:] + array4[:] + array5[:]
train3 = array1[:] + array2[:] + array4[:] + array5[:]
train4 = array1[:] + array2[:] + array3[:] + array5[:]
train5 = array1[:] + array2[:] + array3[:] + array4[:]

val1 = array1[:]
val2 = array2[:]
val3 = array3[:]
val4 = array4[:]
val5 = array5[:]

traindata = [train1, train2, train3, train4, train5]
valdata = [val1, val2, val3, val4, val5]

theta = [0.5,0.5,0.5,0.5]
list_theta = [theta[:] for i in range(5)]
dtheta = [0.5,0.5,0.5,0.5]
bias = 0.5
list_bias = [bias for i in range(5)]
dbias = 0

acc_train = []
acc_val = []

predict_train = []
predict_val = []
errors_train = []
errors_val = []

error_train_final = []
error_val_final = []

acc_train_final = []
acc_val_final = []




errors_train

def Result(x,j):
  res = 0
  for i in range(4):
    global list_theta
    res += (float(x[i])*list_theta[j][i])
    
  global bias
  return res + bias

def Sigmoid(res):
  return 1/(1+math.exp(-res))

Sigmoid(5.6)



def Predict(act):
  if(act>0.5):
    return 1
  else:
    return 0

def Error(type,act):
  return math.pow((type-act),2)

def DthetaUpdate(x,trg,act):
  global dtheta
  for i in range(4):
    value=float(x[i])
    dtheta[i] = 2 *value * (trg-act) * (1-act) * act
    
    

arr=array1[0]
DthetaUpdate(arr,array1[0][5],Sigmoid(5.6))

# dtheta

def DbiasUpdate(trg,act):
  global dbias
  dbias = 2 * (trg-act) * (1-act) * act
 

DbiasUpdate(array1[0][5],Sigmoid(5.6))

dbias

def ThetaUpdate(lr,j):
  global list_theta
  for i in range(4):
    list_theta[j][i] = list_theta[j][i] + (lr*dtheta[i])

def BiasUpdate(lr,j):
  global list_bias
  list_bias[j] += (lr*dbias)



for i in range(99):
  sum_err_train = 0
  sum_err_val = 0 
  sum_acc_train = 0
  sum_acc_val = 0 
  
  for j in range(5):
    sun = 0
    sun2 = 0
    tptn = 0
    tptn2 = 0
    for k in range(19):
    #   train
      dat=traindata[j][k]
      act = Sigmoid(Result(dat,j))

    #   predict_train.append(Predict(act))
      pred = Predict(act)

      if(pred==traindata[j][k][5]):
        tptn+=1

      
      sun += Error(traindata[j][k][5],act)

      DthetaUpdate(traindata[j][k][0:4],float(traindata[j][k][5]),act)
      DbiasUpdate(traindata[j][k][5],act)

      ThetaUpdate(0.8,j)
      BiasUpdate(0.8,j)
 

    sum_err_train += sun/80 
    sum_acc_train += (tptn/80)*100
    
  
    for k in range(19):
   
      act = Sigmoid(Result(valdata[j][k],j))
      pred = Predict(act)

      if(pred==valdata[j][k][5]):
        tptn2+=1
      print()
      sun2 += Error(valdata[j][k][5],act)
    sum_err_val += sun2/20
    sum_acc_val += (tptn2/20)*100
  
  error_train_final.append(sum_err_train/5)
  error_val_final.append(sum_err_val/5)
  acc_train_final.append(sum_acc_train/5)
  acc_val_final.append(sum_acc_val/5)


plt.figure(1)
plt.plot(acc_train_final,'r-', label='train')
plt.plot(acc_val_final,'y-', label='validasi')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc='upper right')

plt.figure(2)
plt.plot(error_train_final,'r-', label='training')
plt.plot(error_val_final,'y-', label='validasi')
plt.xlabel('epoch')
plt.ylabel('error')
plt.legend(loc='upper left')
plt.show()
