#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
file=pd.read_csv("C:/Users/user/Data Science/Social_Network_Ads.csv")
df=pd.DataFrame(file)
df


# In[4]:


df=pd.get_dummies(df)
df


# In[5]:


x=df.drop(['Purchased'],axis=1)
x


# In[6]:


y=df['Purchased'].values
y


# In[7]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# In[8]:


#scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[9]:


#Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(x_train,y_train)


# In[10]:


#predicting the test set results
y_pred=classifier.predict(x_test)


# In[11]:


#to measure the accuracy of model
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy_score(y_test,y_pred)


# In[13]:


#making confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
confusion_matrix(y_test,y_pred)


# In[14]:


#classification report
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
z=classification_report(y_test,y_pred)
print(z)


# In[15]:


#for computing Receiver operating characteristics
from sklearn.metrics import roc_curve
#for computing area under curve
from sklearn.metrics import roc_auc_score


# In[16]:


#visualizing the ROC-AUC curve
y_proba=classifier.predict_proba(x_test)

#we take the predicted values of class 1
y_predicted=y_proba[:,1]

#we check to see if the right values have been considered from the predicted values
print(y_predicted)


# In[17]:


#using roc_curve() to generate fpr & tpr values
fpr,tpr,thresholds=roc_curve(y_test,y_predicted)


# In[19]:


#passing the fpr&tpr values to auc()to calculate the area under curve
from sklearn.metrics import auc
roc_auc=auc(fpr,tpr)
print("Area under the curve for first model :",roc_auc)


# In[20]:


#plotting the ROC curve
import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr,tpr,color='orange',lw=2,label='ROC curve(area under curve=%0.2f)'%roc_auc)
plt.plot([0,1],[0,1],color='darkgrey',lw=2,linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('False Positive Rate(1-Specificity)')
plt.ylabel('True Positive Rate(Sensitivity)')


# In[ ]:




