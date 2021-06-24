#!/usr/bin/env python
# coding: utf-8

# In[5]:


import tensorflow as tf

a= tf.Variable(1, name="a")
b= tf.Variable(2, name="b")
f= a+b

init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as s:
    init.run()
    print ( f.eval() )


# In[ ]:




