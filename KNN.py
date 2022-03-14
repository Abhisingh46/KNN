#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib.inline', '')


# In[9]:


irisData=load_iris()


# In[13]:


x=irisData.data


x.shape


# In[14]:


y=irisData.target
y.shape


# In[15]:


irisData.keys()


# In[16]:


irisData.feature_names


# In[17]:


irisData.target_names


# In[19]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=5)


# In[21]:


knn=KNeighborsClassifier(n_neighbors =4)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
print(y_pred)


# In[22]:


cn=confusion_matrix(y_test,y_pred)
print(cn)


# In[ ]:


#Accuracy = 29/30= 93.33%


# In[23]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
print(acc*100)


# In[24]:


k_range=range(2,20)


# In[28]:


k_score=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    score=cross_val_score(knn,x,y,cv=5,scoring='accuracy')
    k_score.append(score.mean())
print(k_score)


# In[29]:


plt.plot(k_range,k_score)
plt.xlabel('k_values')
plt.ylabel('cross validated Mean Accuracy')
plt.show()


# In[ ]:


#k value will be = 5,6,7,10,11,12
#will not select those value which is divided by number of target variable(3)
#so 7 will be best because of computational cost.


# In[30]:


knn=KNeighborsClassifier(n_neighbors =7)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
print(y_pred)


# In[31]:


cn=confusion_matrix(y_test,y_pred)
print(cn)


# In[ ]:





# In[ ]:





# In[ ]:




