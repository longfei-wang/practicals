
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error as mse
from rdkit import Chem as chem
import itertools as it


# In[14]:


"""
Read in train and test as Pandas DataFrames
"""
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

#store gap values
target_data = df_train.gap.values
#row where testing examples start
test_idx = df_train.shape[0]
#delete 'Id' column

test_data = df_test.drop(['Id'], axis=1)

#clean all data
train_smiles = df_train['smiles']
train_data = df_train.drop(['gap'],axis=1)

# m = chem.MolFromSmiles(train_smiles[0])
# smiles_x_train[0]
# m = chem.AddHs(m)
# print len(m.GetAtoms())
# a = m.GetAtomWithIdx(0)
# a.GetSymbol()


# In[ ]:

from rdkit.Chem.AtomPairs import Pairs
#convert smiles string to bag of bonds dictionary
def smiles2bob(listofsmiles):
    for smiles in listofsmiles:
        m = chem.MolFromSmiles(smiles)
        m = chem.AddHs(m)
        fp = Pairs.GetAtomPairFingerprint(m)
        yield fp.GetNonzeroElements()

def smiles2bob2(smiles):
    m = chem.MolFromSmiles(smiles)
    m = chem.AddHs(m)
    fp = Pairs.GetAtomPairFingerprint(m)
    return fp.GetNonzeroElements()

feature = list()
for i in smiles2bob(train_smiles[:100000]):
    for k,v in i.iteritems():
        if k not in feature:
            feature.append(k)

print len(feature)

#creat a generator that yield a list of elements from smiles string. !treat [Si] as one element
def smiles2list(smiles_list):
    for smiles in smiles_list:
        return_list = list()
        stop_sign = -1
        for i,j in enumerate(smiles):
            if j == '[':
                stop = smiles.find(']',i)
                return_list.append(smiles[i:stop+1])
                stop_sign = stop
            else:
                if stop_sign == -1:
                    return_list.append(smiles[i])
                elif i == stop_sign:
                    stop_sign = -1
        yield return_list

#break down features like languague learning, adjacent words/elements

mygenerator = smiles2list(train_smiles[:100000])
features2 = list()

for smiles in mygenerator:
    for j in range(len(smiles)):
        #single char double char tri char ....
        for x in [smiles[j],''.join(smiles[j:j+2]),''.join(smiles[j:j+3]),''.join(smiles[j:j+4])]:
            if x not in features2: 
                if ')' not in x: # ')' means end of a subgroup, not much information there
                    features2.append(x)

print len(features2)

#generate feature matrix from the features collected. 
def transform_features(smiles):
    dbob = smiles2bob2(smiles)
    return ([dbob.get(i,0) for i in feature] + [smiles.count(i) for i in features2])


new_train_data = map(transform_features,train_smiles)


#np.save('train_data',new_train_data)
#new_train_data = np.load('train_data.npy')


#0.5% data is kept for testing purpose

x_train, x_test, y_train, y_test = cross_validation.train_test_split(new_train_data, target_data, test_size=0.05, random_state=0)
print "Train features:", x_train.shape
print "Train gap:", y_train.shape
print "Test features:", x_test.shape
print y_test.shape


# In[12]:

import time
start_time = time.time()

RF = RandomForestRegressor(n_estimators=80,n_jobs=-1,max_depth=1000,max_features=.8,bootstrap=True)
RF.fit(x_train, y_train)
RF_pred = RF.predict(x_test)

print("--- %s seconds ---" % (time.time() - start_time))
print np.sqrt(mse(y_test,RF_pred))





new_df_test = map(transform_features,df_test['smiles'])



new_pred = RF.predict(new_df_test)




def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")



#write_to_file("longfei2.csv", new_pred)




