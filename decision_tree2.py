#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import tree

input_file= "C:/Users/ABIR/Desktop/PastHires.csv"
df=pd.read_csv(input_file, header = 0)


# In[2]:


df.head()


# In[3]:


d = {'Y': 1, 'N':0}
df['Hired']=df['Hired'].map(d)
df['Employed?']=df['Employed?'].map(d)
df['Top-tier school']=df['Top-tier school'].map(d)
df['Interned']=df['Interned'].map(d)
d={'BS':0, 'MS':1, 'PhD': 2}
df['Level of Education'] = df['Level of Education'].map(d)
df.head()


# In[4]:


features = list(df.columns[:6])
features


# In[5]:


y= df["Hired"]
X=df[features]
clf=tree.DecisionTreeClassifier()
clf = clf.fit(X,y)


# In[11]:


from IPython.display import Image
from sklearn.externals.six import StringIO
import pydot

dot_data=StringIO()
tree.export_graphviz(clf,out_file=dot_data,feature_names=features)
(graph,)=pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


# In[ ]:




