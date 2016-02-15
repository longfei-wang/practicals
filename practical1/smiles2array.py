# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 23:14:59 2016

@author: jerome
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import StratifiedKFold
import numpy as np

"""
Read in train and test as Pandas DataFrames
"""
loc = "/Users/jerome/Desktop/HES/COMPSCI-181/Practicals/Practical 1/"
df_train = pd.read_csv(loc+"train.csv")
df_test = pd.read_csv(loc+"test.csv")

#store gap values
Y_train = df_train.gap.values
#delete 'Id' column
df_train = df_train.smiles.values
#delete 'gap' column
df_test = df_test.smiles.values

test_idx = df_train.shape[0]

X_train = []
for iidx in range(len(df_train)):
    tempArr = []
    for jidx in range(len(df_train[iidx])):
        tempArr.append(ord(df_train[iidx][jidx]))
    X_train.append(tempArr)
    print iidx

df_X_train = pd.DataFrame(X_train)

X_test = []
for iidx in range(len(df_test)):
    tempArr = []
    for jidx in range(len(df_test[iidx])):
        tempArr.append(ord(df_test[iidx][jidx]))
    X_test.append(tempArr)
    print iidx

df_X_test = pd.DataFrame(X_test)

df_all = pd.concat((df_X_train, df_X_test), axis=0)

df_all = df_all.fillna(0).values

X_train_ASCII = df_all[:test_idx]
X_test_ASCII = df_all[test_idx:]
        
RF = RandomForestRegressor()
kfold = StratifiedKFold(y=np.array(Y_train), n_folds=10, random_state=1)
scores = []
for k, (train, test) in enumerate(kfold):
    RF.fit(X_train_ASCII[train], Y_train[train])
    print np.mean(RF.score(X_train_ASCII[test], Y_train[test]))

RFA_pred = RF.predict(X_test_ASCII)

def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")

write_to_file(loc+"WWHAN_021101_ASCII_RF.csv", RFA_pred)
