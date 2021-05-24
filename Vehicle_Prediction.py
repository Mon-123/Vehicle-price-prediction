#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd


# In[2]:


os.chdir("C:\\Users\\Monali\\OneDrive\\Desktop\\krush naik")


# In[3]:


os.getcwd()


# # Importing dataset
# 

# In[4]:


Data = pd.read_csv('car data.csv')


# In[5]:


Data.head()


# In[6]:


Data.shape


# In[7]:


Data.describe()


# In[8]:


Data.info()


# In[9]:


Data.isnull().sum()


# In[10]:


Data['Present Year'] = 2020


# In[11]:


Data.head()


# In[12]:


Data['Years'] = Data['Present Year']-Data['Year']


# In[13]:


Data.head()


# In[14]:


Data.drop(['Year'],axis =1, inplace = True)


# In[15]:


Data.drop(['Present Year'],axis=1,inplace=True)


# In[16]:


Data.drop(['Car_Name'],axis = 1, inplace= True)


# In[17]:


Data.head()


# In[18]:


#converting categorical features in one hot encoded
Data = pd.get_dummies(Data,drop_first=True)


# In[19]:


Data.head()


# In[20]:


import seaborn as sns


# In[21]:


sns.pairplot(Data)


# # Finding the corelation

# In[22]:


cormat=Data.corr()


# In[23]:


top_corr_features=cormat.index


# In[24]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


cormat=Data.corr()
top_corr_features=cormat.index
plt.figure(figsize=(20,20))


# In[26]:


#plot heatmap


# In[27]:


g=sns.heatmap(Data[top_corr_features].corr(),annot=True,cmap="Blues_r")


# In[28]:


x=Data.iloc[:,1:]
y=Data.iloc[:,0]


# In[29]:


x.head()


# In[30]:


y.head()


# identify important features, it can be done by ExtraTreesRegressor

# In[31]:


from sklearn.ensemble import ExtraTreesRegressor 


# In[32]:


model = ExtraTreesRegressor()


# In[33]:


model.fit(x,y)


# In[34]:


print(model.feature_importances_)


# In[35]:


#plot graph of feature importance for better visualization
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# In[36]:


from sklearn.model_selection import train_test_split


# In[37]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)


# In[38]:


x_train.shape


# In[39]:


model.fit(x_train,y_train)


# In[40]:


from sklearn.ensemble import RandomForestRegressor


# In[41]:


rf_random = RandomForestRegressor()


# # Hyperameters

# In[84]:


#randomized search cv - helps to find out the best parameters...

#number of trees in random forest
#n_estimators = ([int(x) for x in np.linspace(start = 100,stop = 1200, num = 12)])
#number of features to consider at every split
max_features = ['auto','sqrt']
#maximum number of levels in tree
max_Depth = [int(x) for x in np.linspace(5, 30, num = 6)]
#minimum number of samples required to split the node
min_samples_split = [2,5,10,15,100]
#minimum numbers of samples required at each sleaf node
min_samples_leaf = [1,2,5,10]


# In[85]:


from sklearn.model_selection import RandomizedSearchCV


# In[86]:


#create random grid - selects the best parameters
random_grid = {'n_estimators' : n_estimators,
              'max_features' : max_features,
              'max_depth' : max_Depth,
              'min_samples_split' : min_sample_splits,
              'min_samples_leaf' : min_sample_leaf}

print(random_grid)


# In[87]:


#create a best model to tune
from sklearn.model_selection import RandomizedSearchCV


# In[88]:


rf = RandomForestRegressor()


# In[89]:


rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, scoring = 'neg_mean_squared_error', n_iter = 10, cv =4, verbose=2, random_state = 42, n_jobs= 1)


# In[90]:


rf_random


# In[91]:


rf_random.fit(x_train, y_train)


# In[93]:


predictions= rf_random.predict(x_test)


# In[94]:


predictions


# In[96]:


sns.distplot(y_test-predictions)


# In[97]:


plt.scatter(y_test, predictions)


# In[ ]:




