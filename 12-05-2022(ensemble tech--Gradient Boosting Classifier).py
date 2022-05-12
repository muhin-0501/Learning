#!/usr/bin/env python
# coding: utf-8

# In[1]:


# file importing

import pandas as pd
file=pd.read_csv("C:/Users/user/Data Science/Social_Network_Ads.csv")
df=pd.DataFrame(file)
df


# In[2]:


# changing as Dummies

df=pd.get_dummies(file)
df


# In[3]:


# x assigning

x=df.drop(['Purchased'],axis=1)
x


# In[4]:


# y assigning

y=df['Purchased'].values
y


# In[5]:


# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# In[6]:


#scaling

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[8]:


from sklearn.ensemble import GradientBoostingClassifier

grad_boost_clf = GradientBoostingClassifier(
                        n_estimators=500, 
                        learning_rate=0.8, 
                        random_state=42,
                        max_depth=2)

grad_boost_clf.fit(x_train, y_train)


# In[14]:


#predicting the test set results
#use x_test

y_pred=grad_boost_clf.predict(x_test)


# In[15]:


#to measure the accuracy of model
# ACCURACY

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy_score(y_test,y_pred)


# In[23]:


#to measure the accuracy of model
# ACCURACY

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
confusion_matrix(y_test,y_pred)


# In[21]:


#to measure the accuracy of model
# ACCURACY

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
z=classification_report(y_test,y_pred)
print(z)


# In[ ]:




