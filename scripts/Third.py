#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Useful starting lines
import numpy as np

# # Load the  data into feature matrix, class labels, and event ids:

# In[2]:


from proj1_helpers import *
#Change the path according to where is your data 
DATA_TRAIN_PATH = r"C:\Users\USER\Desktop\MA1\ML\train.csv" 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)


# In[3]:


DATA_TEST_PATH = r"C:\Users\USER\Desktop\MA1\ML\test.csv" # TODO: download train data and supply path here 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)


# In[4]:


### Define variables for the dimensions of the data
N = tX.shape[0]
D = tX.shape[1]

#the number of the feature that contains the pri_jet_number
jet_num_col=22


# # Data Pre-Processing

# Because we will partition the data, and then add/delete features, we will fix the column of the jet_num_col to the first column

# In[6]:


# swapping the column with index of
# original array
tX[:, [jet_num_col, 0]] = tX[:, [0, jet_num_col]]
tX_test[:, [jet_num_col, 0]] = tX_test[:, [0, jet_num_col]]


# In[7]:


jet_num_col=0


# ## Def func

# In[8]:


### Transform tX by changing -999 with the mean of column
def transformTX(tX):
    tX2 = np.copy(tX)
    tX2[tX2 == -999] = 0
    means = np.mean(tX2, axis=0)
    for i in range(N):
        for j in range(D):
            if tX[i][j] == -999:
                tX[i][j] = means[j]
    return tX


# In[9]:


### Apply log to smoothen data
def maybeAddLog(tX):
    tX2 = np.copy(tX)
    mins = np.min(tX2, axis=0)
    for i in range(D):
        if mins[i]>0:
            for k in range(N):
                tX[k][i] = np.log(tX[k][i])
    return tX


# In[10]:


### Features Expansion to capture non linear data
def featuresExpansion(tX, degree):
    res = np.zeros(N).reshape(-1,1)
    for i in range(D):
        for d in range(1,degree+1):
            col = tX[:,i]**d
            col = col.reshape(-1,1)
            res = np.hstack((res, col))
    res = np.delete(res, 0,1)
    return res


# ## Partitioning the data

# In[11]:


#div data is a table (that will be changed into a numpy) that will contains the datasets according to their pri_jet_num
div_x= [[],[],[]]
div_y=[[],[],[]]
div_ids=[[],[],[]]


# In[12]:


# tX[i] gives you the ith row, therefore the ith data point
# tX[:,j] gives you the jth feature 


# In[13]:


#For every data point
for i in range (N):
    #we extract its pri_jet_num
    pri_jet_num=int(tX[i,jet_num_col])
    #we assign it to the new data set based on its pri_jet_num
    if (pri_jet_num in [0,1]): 
        div_x[pri_jet_num].append(tX[i])
        div_y[pri_jet_num].append(y[i])
        div_ids[pri_jet_num].append(ids[i])
        
    if (pri_jet_num in [2,3]):
        div_x[2].append(tX[i]) 
        div_y[2].append(y[i])
        div_ids[2].append(ids[i])       


# In[14]:


#transforming all resultst to numpy arrays
for i in range(3):
    div_x[i]=np.array(div_x[i])
    div_y[i]=np.array(div_y[i])
    div_ids[i]=np.array(div_ids[i])

div_x=np.array(div_x)
div_y=np.array(div_y)
div_ids=np.array(div_ids)


# In[15]:


div_x_test= [[],[],[]]
div_id_test=[[],[],[]]
N_test=tX_test.shape[0]
#For every data point
for i in range (N_test):
    #we extract its pri_jet_num
    pri_jet_num=int(tX_test[i,jet_num_col])
    #we assign it to the new data set based on its pri_jet_num
    if (pri_jet_num in [0,1]): 
        div_x_test[pri_jet_num].append(tX_test[i])
        div_id_test[pri_jet_num].append( ids_test[i])
    if (pri_jet_num in [2,3]):
        div_x_test[2].append(tX_test[i])
        div_id_test[2].append( ids_test[i])


#transforming all resultst to numpy arrays
for i in range(3):
    div_x_test[i]=np.array(div_x_test[i])

div_x_test=np.array(div_x_test)


# ## Deleting Features

# In[18]:


#del_features[i] is False if the feature will be deleated

#del_features[i] is for the ith dataset
#del_features[:j] is for the jth feature

del_features= [[],[],[]]


# ### Undefined Features

# In[19]:


#takes as input a list of lists called del_features
#del_features[i] contains a list of the indices of the features to be deleted in ith dataset 
def f_del_features(del_features):
    for i in range(3):
        div_x[i]=np.delete(div_x[i],del_features[i],axis=1)
        div_x_test[i]=np.delete(div_x_test[i],del_features[i],axis=1)


# In[20]:


# We add all the features that contain only one values
for i in range (3):
    for j in range(D):
        s= set(div_x[i][:,j])
        if (len(s)<2):
            del_features[i].append(j)


# In[22]:


f_del_features(del_features)
#We reinitialize del_features
del_features= [[],[],[]]


# ### Correlated Features

# In[24]:


corrs=[[],[],[]]
for i in range(3):
    corrs[i]= np.corrcoef(div_x[i].T)


# In[26]:


for k in range(3):
    corr=corrs[k]
    D_k=div_x[k].shape[1]
    for i in range(1,D_k):
        for j in range(i+1,D_k):
            if corr[i,j] >= 0.9:
                if j not in del_features[k]:
                    del_features[k].append(j)                 


# In[28]:


f_del_features(del_features)
#We reinitialize del_features
del_features= [[],[],[]]


# ## TO ADD

# In[30]:


# tX[i] gives you the ith row, therefore the ith data point
# tX[:,j] gives you the jth feature 

def rescaling(tX):
    tX=tX.astype(float)
    print(tX.shape)
    N=tX.shape[0]
    D= tX.shape[1]
    for n_feat in range(D):
        print(n_feat)
        print()
        data=tX[n_feat].astype(float)
        print(data)
        print((data - np.min(data)))
        tX[n_feat]=(data - np.min(data)) / (np.max(data) - np.min(data))
    return tX


# ## All Together

# In[31]:


### Transform tX by changing -999 with the mean of column
def transformTX(tX):
    N=tX.shape[0]
    D=tX.shape[1]
    tX2 = np.copy(tX)
    tX2[tX2 == -999] = 0
    means = np.mean(tX2, axis=0)
    for i in range(N):
        for j in range(D):
            if tX[i][j] == -999:
                tX[i][j] = means[j]
    return tX


# In[32]:


### Apply log to smoothen data
def maybeAddLog(tX):
    N=tX.shape[0]
    D=tX.shape[1]
    tX2 = np.copy(tX)
    mins = np.min(tX2, axis=0)
    for i in range(D):
        if mins[i]>0:
            for k in range(N):
                tX[k][i] = np.log(tX[k][i])
    return tX


# In[33]:


### Features Expansion to capture non linear data
def featuresExpansion(tX, degree):
    N=tX.shape[0]
    D=tX.shape[1]
    res = np.zeros(N).reshape(-1,1)
    for i in range(D):
        for d in range(1,degree+1):
            col = tX[:,i]**d
            col = col.reshape(-1,1)
            res = np.hstack((res, col))
    res = np.delete(res, 0,1)
    return res


# In[34]:


deg = 7
for i in range(3):
    div_x[i] = transformTX(div_x[i])
    div_x[i] = maybeAddLog(div_x[i])
    div_x[i] = featuresExpansion(div_x[i], deg)
    
    
    div_x_test[i] = transformTX(div_x_test[i])
    div_x_test[i] = maybeAddLog(div_x_test[i])
    div_x_test[i] = featuresExpansion(div_x_test[i], deg)


# # Linear Regression

# Remark: We are assuming that we DO NOT have an offset and that w = {w1, w2, ... , wD} where D=30 in our case

# In[41]:


### Least squares regression using normal equations
def least_squares(y, tX):
    return np.linalg.solve(tX.T@tX,tX.T@y)


# # Generate predictions and save ouput in csv format for submission:

# In[63]:


OUTPUT_PATH = r"C:\Users\USER\Desktop\MA1\ML\output.csv" # TODO: fill in desired name of output file for submission


# In[64]:


###IF WE ARE USING LEAST_SQUARES 
weights= [[],[],[]]
pred=  [[],[],[]]
for i in range(3):
    weights[i] = least_squares(div_y[i],div_x[i])
    pred[i] = predict_labels(weights[i],div_x_test[i] )


# In[66]:


y_pred= []
ctr= [0,0,0]
for i in range(N_test):
    pri_jet_num=int(tX_test[i,jet_num_col])
    if (pri_jet_num in [0,1]): 
        idx= ctr[pri_jet_num]
        ctr[pri_jet_num]+=1
        x= pred[pri_jet_num][idx]
        y_pred.append(x)
    if (pri_jet_num in [2,3]):
        idx= ctr[2]
        ctr[2]+=1
        x= pred[2][idx]
        y_pred.append(x)
    


# In[68]:


#Create submission
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

