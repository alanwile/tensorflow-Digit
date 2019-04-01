#!/usr/bin/env python
# coding: utf-8

# In[8]:


import tensorflow as tf 
from tensorflow import keras

import numpy as nump
import matplotlib.pyplot as plt


# In[9]:


dig = keras.datasets.mnist
(x, y ),  (x_prueba , y_prueba) = dig.load_data()


# In[10]:


plt.figure()
plt.imshow(x[100])
plt.show()


# In[11]:


x.shape


# In[12]:


x=x/255.0
x_prueba=x_prueba/255.0


# In[13]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.sigmoid),
    keras.layers.Dense(10, activation=tf.nn.sigmoid)
])


# In[15]:


model.compile(optimizer='adam',  loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x, y, epochs=20)
model.evaluate(x_prueba,y_prueba)

predicciones = model.predict(x_prueba)
def predecir(xx):
    plot.figure()
    plot.imshow(x_prueba[xx])
    plot.xlabel(y_prueba[xx])


# In[16]:


len(y_prueba)


# In[22]:


predecir(11)


# In[23]:


test_loss, test_acc = model.evaluate(x_prueba, y_prueba)


# In[34]:


predecir(4)


# In[ ]:




