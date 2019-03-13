#!/usr/bin/env python
# coding: utf-8

# ## problem statement here is to predict house prices based on different varibales like location, distance, rate of security

# ## import all libraries to use

# In[26]:


import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split


# ### load the dataset of boston from sklearn library 

# In[11]:


from sklearn.datasets import load_boston
boston = load_boston()


# ##### divide data into two dataframes by the use of pandas

# In[14]:


df_x = pd.DataFrame(boston.data,columns=boston.feature_names)
df_y = pd.DataFrame(boston.target)


# In[18]:


#provide a summary description of our data
df_x.describe()


# applying regression

# In[21]:


reg = linear_model.LinearRegression()


# splitting our data in train and test data

# In[28]:


x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.2,random_state=4)


# In[30]:


#fit the dataset into regression function

reg.fit(x_train,y_train)


# fit on coefficient
# 

# coefficience gives us an idea of how much the dependent variable will increase if the independent variable value increases by one (1) 

# In[32]:


reg.coef_


# In[52]:


#make predictions
a = reg.predict(x_test)


# In[48]:


a[6]


# In[51]:


#check mean error
#the lesser the mean error the better the prediction
np.mean((a-y_test)**2)

