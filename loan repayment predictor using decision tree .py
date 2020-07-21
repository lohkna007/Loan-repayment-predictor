#!/usr/bin/env python
# coding: utf-8

# # collecting data and libraries

# In[84]:


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import seaborn as sns
from sklearn.metrics import confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')
import math


# In[14]:


loan_repay = pd.read_csv("/home/gaurav/Downloads/loan_data_final.csv" ,sep = ",", header =0)


# In[15]:


loan_repay.head(5)


# In[18]:


print("length:" + str(len(loan_repay)))


# In[22]:


sns.countplot(x = "not.fully.paid", data =loan_repay)


# In[24]:


sns.countplot(x = "not.fully.paid",hue = "purpose", data =loan_repay)


# In[28]:


loan_repay["installment"].plot.hist(bins = 5, figsize =(10,5))


# # Data Wranling

# In[67]:


loan_repay.info()
loan_repay.isnull().sum()


# In[68]:


loan_repay.dropna(inplace = True )


# In[69]:


loan_repay.drop("credit.policy", axis = 1, inplace = True)


# In[70]:


loan_repay.head(5)


# # Train Dataset

# In[76]:


x = loan_repay.values[:, 1:11]
y = loan_repay["not.fully.paid"]


# In[127]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.30, random_state = 30)


# In[128]:


classify_Entropy = DecisionTreeClassifier(criterion = "entropy", random_state= 30 )


# In[129]:


classify_Entropy.fit(x_train, y_train)


# # prediction and accuracy check

# In[130]:


prediction = classify_Entropy.predict(x_test)


# In[131]:


confusion_matrix(y_test,prediction)


# In[132]:


accuracy_score(y_test,prediction)*100   


# In[ ]:




