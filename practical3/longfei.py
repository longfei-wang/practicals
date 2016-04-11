
# coding: utf-8

# In[2]:

import numpy as np
import warnings



# In[3]:

artist_matrix = np.load('artist_matrix2.npy').item()
print len(artist_matrix.values()[0])


# In[38]:

#artist_table = np.load('table1.npy').item()


# In[36]:



# In[103]:

#n = neigh.kneighbors(np.array(artist_matrix['03098741-08b3-4dd7-b3f6-1b0bfa2c879c']),5,return_distance=True)



# In[50]:

#user_matrix = np.load('user_matrix1.npy').item()



# # In[91]:

# uids = []
# udata = []

# for k,v in user_matrix.iteritems():
#     uids.append(k)
#     udata.append(v)

# uX = np.array(udata)
# print uX.shape


# # In[92]:

# uneigh = NearestNeighbors()
# uneigh.fit(uX)


# In[99]:

#n = uneigh.kneighbors(np.array(user_matrix['5b246febe3d5d4efe8d632b974dbaf4bba47a3a4']),5,return_distance=True)




# In[123]:



# In[ ]:

######################################################################################
import csv
import random
train_file = 'train.csv'
train_data = {}
test_data = {}

with open(train_file, 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    for row in train_csv:

        user   = row[0]
        artist = row[1]
        plays  = int(row[2])
        
        if random.random() < 0.01:
            if not user in test_data:
                test_data[user] = {}

            test_data[user][artist] = plays
        else:

            if not user in train_data:
                train_data[user] = {}

            train_data[user][artist] = plays

print len(train_data.keys())

#user_medians = np.load('user_medians.npy').item()

# def predict(user,artist,nu,na):
#     s_artists = neigh.kneighbors(np.array(artist_matrix[artist]),na,return_distance=True)
#     s_users = uneigh.kneighbors(np.array(user_matrix[user]),nu,return_distance=True)
    
#     uuu = [(uids[i],idist) for i,idist in zip(s_users[1][0],s_users[0][0])]
#     aaa = [(ids[j],jdist) for j,jdist in zip(s_artists[1][0],s_artists[0][0])]
    
#     return uuu,aaa


# def get_score(quser,qartist,nu=200,na=10):


#     result = predict(quser,qartist,nu,na)

#     score = list()
#     n = 0 
#     for j,jdist in result[1]:
#         for i,idist in result[0]:
#             if not (i==quser and j==qartist):
#                 try:
#                     score.append(train_data[i][j])
#                     #print train_data[i][j],idist, jdist
#                     n += 1
#                 except:
#                     pass
                
#                 if n > 5:
#                     return score
    
#     if len(score) != 0:
#         return score
#     elif nu < 1000:
#         return get_score(quser,qartist,nu*2,na*2)
#     else:
#         return user_medians[quser]

warnings.filterwarnings('ignore')

plays_array  = []
user_medians = {}

artist_plays = {}

for user, user_data in train_data.iteritems():
    user_plays = []
    for artist, plays in user_data.iteritems():
        plays_array.append(plays)
        user_plays.append(plays)

        try:
            artist_plays[artist] += [plays]
        except:
            artist_plays[artist] = [plays]

    user_medians[user] = np.median(np.array(user_plays))
global_median = np.median(np.array(plays_array))


artist_medians = {}
for artist, plays in artist_plays.iteritems():
    artist_medians[artist] = np.median(plays)

artist_mean = np.mean(artist_medians.values())

#print artist_plays

ids = list()
data = list()
for k,v in artist_matrix.iteritems():
    ids.append(k)
    data.append(v)

X = np.array(data)
print X.shape


# In[41]:

from sklearn.neighbors import NearestNeighbors

neigh = NearestNeighbors()
neigh.fit(X)



distance_matrix = dict()
for i in ids:
    distance_matrix[i] =  neigh.kneighbors(np.array(artist_matrix[i]),10,return_distance=False)[0]


def get_median(user,artist):
    neighbours = distance_matrix[artist]

    numPlays = []

    for i in neighbours:
        name = ids[i]

        try:
            numPlays.append(train_data[user][name])
            
        except:
            pass

    if len(numPlays) == 0:
        x = user_medians[user]
    else:
        x = np.median(numPlays)


    return x




for p in [1,5,10,20,30,40,50]:
    n = 0
    error = 0.0
    std_error = 0.0

    for k, v in test_data.iteritems():
        for k1,v1 in v.iteritems():
            n +=1

            #x = get_median(k,k1)
            x = user_medians[k] + (artist_medians[k1] - artist_mean)/float(8) + (get_median(k,k1) - user_medians[k])/float(20)

            error += np.abs(x - v1)
            
            std = user_medians[k]

            std_error += np.abs(std - v1)
            
    
    print (error/float(n) - std_error/float(n)), n




# test_file  = 'test.csv'
# soln_file = 'longfei1.csv'

# with open(test_file, 'r') as test_fh:
#     test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
#     next(test_csv, None)

#     with open(soln_file, 'w') as soln_fh:
#         soln_csv = csv.writer(soln_fh,
#                               delimiter=',',
#                               quotechar='"',
#                               quoting=csv.QUOTE_MINIMAL)
#         soln_csv.writerow(['Id', 'plays'])

#         n = 0
#         for row in test_csv:
#             #n += 1
#             #print n

#             id     = row[0]
#             user   = row[1]
#             artist = row[2]

#             soln_csv.writerow([id, get_median(user,artist)])

