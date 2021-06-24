#!/usr/bin/env python
# coding: utf-8

# In[20]:


import math
import numpy as np
import pandas as pd
import pandas_datareader as pdd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt


# In[21]:


df = pdd.DataReader('AAPL', data_source='yahoo', start='2013-01-01', end='2019-12-30')
df


# In[22]:


plt.figure(figsize=(16,8))
plt.title('close price movement')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('close price in $', fontsize=18)
plt.show


# In[23]:


data = df.filter(['Close'])
dataset= data.values
len(dataset)


# In[24]:


training_data_size = math.ceil(len(dataset)*.7)
training_data_size


# In[25]:


scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data


# In[26]:


train_data = scaled_data[0:training_data_size, :]
x_train = []
y_train = []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i,0])
    if i<=60:
        print(x_train)
        print(y_train)


# In[27]:


x_train, y_train = np.asarray(x_train), np.asarray(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape


# In[28]:


model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# In[29]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[30]:


model.fit(x_train, y_train, batch_size=1, epochs=1)


# In[31]:


test_data = scaled_data[training_data_size -60: ,:]
x_test = []
y_test = dataset[training_data_size:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i ,0])


# In[32]:


x_test = np.array(x_test)


# In[33]:


x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))


# In[35]:


predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# In[37]:


rmse = np.sqrt(np.mean(predictions -y_test)**2)
rmse


# In[38]:


train= data[:training_data_size]
valid = data[training_data_size:]
valid['predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Model LM')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close price in $', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'predictions']])
plt.legend(['Train', 'Val', 'predictions'],loc='lower right')
plt.show


# In[39]:


valid


# In[ ]:




