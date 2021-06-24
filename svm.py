#!/usr/bin/env python
# coding: utf-8

# In[12]:


#create fake income/age clusters for N people in k clusters
import random
from numpy import *
import numpy as np
from matplotlib import pyplot as plt
def createClusteredData(N, k):
    pointsPerCluster = float(N)/k
    x = []
    y = []
    for i in range (k):
        incomeCentroid = random.uniform(20000.0, 200000.0)
        ageCentroid = random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            x.append([np.random.normal(incomeCentroid, 10000.0), np.random.normal(ageCentroid, 2.0)])
            y.append(i)
    x=np.array(x)
    y=np.array(y)
    return x, y


# In[13]:


(X, y) = createClusteredData(100, 5)

plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=y.astype(np.float))
plt.show()


# In[20]:


from sklearn import svm, datasets

C = 1.0
svc= svm.SVC(kernel='linear', C=C).fit(X, y)


# In[23]:


def plotPredictions(clf):
    xx, yy = np.meshgrid(np.arange(0, 250000, 10), np.arange(10, 70, 0.5))
    z= clf.predict(np.c_[xx.ravel(), yy.ravel()])
    plt.figure(figsize=(8,6))
    z=z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:,0], X[:,1], c=y.astype(np.float))
    plt.show()

plotPredictions(svc)

    


# In[31]:


temp=svc.predict([20000, 40])
temp=temp.reshape(-1,1)


# In[30]:


y=svc.predict([50000, 65])
#print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
print (np.concatenate(y.reshape(-1,1)))


# In[ ]:




