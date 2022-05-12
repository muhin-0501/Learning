#!/usr/bin/env python
# coding: utf-8

# In[2]:


# file importing

import pandas as pd
file=pd.read_csv("C:/Users/user/Data Science/Social_Network_Ads.csv")
df=pd.DataFrame(file)
df


# In[6]:


# changing as Dummies

df=pd.get_dummies(file)
df


# In[25]:


# x assigning

x=df.drop(['Purchased'],axis=1)
x


# In[26]:


# y assigning

y=df['Purchased'].values
y


# In[66]:


# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# In[67]:


#scaling

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[68]:


#Fitting Bagging Classifier

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_classifier = BaggingClassifier(
      DecisionTreeClassifier(class_weight='balanced'),
    max_samples=0.5, max_features=0.5, bootstrap=False
)

bag_classifier.fit(x_train, y_train)


# In[69]:


#predicting the test set results
#use x_test

y_pred=bag_classifier.predict(x_test)


# In[70]:


#to measure the accuracy of model
# ACCURACY

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy_score(y_test,y_pred)


# In[71]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
confusion_matrix(y_test,y_pred)


# In[73]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
x=classification_report(y_test,y_pred)
print(x)


# In[ ]:




