
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error as mse


# In[2]:

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

#0.5% data is kept for testing purpose

x_train, x_test, y_train, y_test = cross_validation.train_test_split(train_data, target_data, test_size=0.05, random_state=0)
print "Train features:", x_train.shape
print "Train gap:", y_train.shape
print "Test features:", x_test.shape
print y_test.shape


# In[3]:

smiles_x_train = x_train['smiles']
smiles_x_test = x_test['smiles']


# In[4]:

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


# In[5]:

#break down features like languague learning, adjacent words/elements

mygenerator = smiles2list(smiles_x_train[:50000])
features = list()

for smiles in mygenerator:
    for j in range(len(smiles)):
        #single char double char tri char ....
        for x in [smiles[j],''.join(smiles[j:j+2]),''.join(smiles[j:j+3]),''.join(smiles[j:j+4])]:
            if x not in features: 
                if ')' not in x: # ')' means end of a subgroup, not much information there
                    features.append(x)

print len(features)


# In[6]:

#generate feature matrix from the features collected. 
def transform_features(smiles):
    return [smiles.count(i) for i in features]

new_x_test = map(transform_features,smiles_x_test)

new_x_train = map(transform_features, smiles_x_train)



# 10 0.072
# 20
# 30 0.066
# 40
# 50 0.065
# 60
# 70
# 80
# 90
# 100 0.064
# 
# 
# all features
# 100 

# In[ ]:
print "regression"
#regression
RF2 = RandomForestRegressor(n_estimators=50,n_jobs=-1,max_depth=500)
RF2.fit(new_x_train, y_train)
RF2_pred = RF2.predict(new_x_test)
print np.sqrt(mse(y_test,RF2_pred))

print "saving"

new_df_test = map(transform_features,df_test['smiles'])
new_pred = RF2.predict(new_df_test)


def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")



#write_to_file("longfei4.csv", new_pred)





