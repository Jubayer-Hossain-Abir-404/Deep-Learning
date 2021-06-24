#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


(X_train, y_train), (X_test, y_test)= datasets.cifar10.load_data()
X_train.shape


# In[3]:


X_test.shape


# In[10]:


y_train[:5]


# In[11]:


y_train = y_train.reshape(-1,)
y_train[:5]


# In[13]:


classes = ["airplane", "automobile", "bird", "cat", "deer", "deer", "frog", "horse", "ship", "truck"]


# In[15]:


classes[2]


# In[16]:


def plot_sample(X, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])


# In[21]:


plot_sample(X_train, y_train, 0)


# In[22]:


plot_sample(X_train, y_train, 3)


# In[25]:


X_train = X_train/255
X_test = X_test /255


# In[26]:


ann = models.Sequential([
        layers.Flatten(input_shape=(32,32,3)),
        layers.Dense(3000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(10, activation='sigmoid')
    ])

ann.compile(optimizer='SGD',
            loss='sparse_categorical_crossentropy',
           metrics=['accuracy'])

ann.fit(X_train, y_train, epochs=5)


# In[27]:


ann.evaluate(X_test, y_test)


# In[28]:


from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(y_test, y_pred_classes))


# In[29]:


cnn = models.Sequential([
    
        #cnn
        #dense
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])


# In[ ]:




