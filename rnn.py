#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb


# In[7]:


print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=20000)


# In[3]:


x_train[0]


# In[8]:


y_train[0]


# In[9]:


x_train = sequence.pad_sequences(x_train, maxlen=80)
x_test = sequence.pad_sequences(x_test, maxlen=80)


# In[10]:


model= Sequential()
model.add(Embedding(20000, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))


# In[13]:


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
             )


# In[14]:


model.fit(x_train, y_train,
          batch_size=32,
          epochs=15,
          verbose=2,
          validation_data=(x_test, y_test)
         )


# In[15]:


score, acc = model.evaluate(x_test, y_test,
                            batch_size=32,
                            verbose=2
                           )
print('Test score:', score)
print('Test accuracy:', acc)


# In[ ]:




