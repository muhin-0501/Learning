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


# In[7]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score

log_classifier =  LogisticRegression()
sv_classifier = SVC()
sgd_classifier = SGDClassifier()


def classifiers(clf1, clf2, clf3, X_train, y_train):
    
    
    # A list of all classifiers
    clfs = [clf1, clf2, clf3]
    
    # An empty list to comprehend 
    all_clfs_acc = []
    
    # Train each classifier, evaluate it on the training set 
    # And append the accuracy to 'all_clfs_acc' 
    
    for clf in clfs:
        
        clf.fit(X_train, y_train)
        preds = clf.predict(X_train)
        acc = accuracy_score(y_train,preds)
        acc = acc.tolist()
        all_clfs_acc.append(acc)
        
    return all_clfs_acc


# In[9]:


from sklearn.ensemble import VotingClassifier

vot_classifier = VotingClassifier(
    
    estimators=[('log_reg', log_classifier),
                ('svc', sv_classifier),
                ('sgd', sgd_classifier)], 
    voting='hard')

vot_classifier.fit(x_train, y_train)


# In[10]:


from sklearn.metrics import accuracy_score

def accuracy(model, data, labels):
    
    predictions = model.predict(data)
    acc = accuracy_score(labels, predictions)
    
    return acc


# In[12]:


#predicting the test set results
#use x_test

y_pred=vot_classifier.predict(x_test)


# In[13]:


#to measure the accuracy of model
# ACCURACY

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy_score(y_test,y_pred)


# In[14]:


#to measure the accuracy of model
# ACCURACY

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
confusion_matrix(y_test,y_pred)


# In[16]:


#to measure the accuracy of model
# ACCURACY

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
y=classification_report(y_test,y_pred)
print(y)


# In[ ]:




