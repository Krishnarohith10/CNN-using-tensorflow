#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as split
import tensorflow as tf


# In[42]:


file = h5py.File('DataSet.h5')
data = file.get('dataset_1')
X = data.get('input')
y = data.get('output')
X, y = X[()], y[()]


# In[43]:


X_train, X_test, y_train, y_test = split(X, y)


# In[44]:


X_train.shape


# In[45]:


def create_placeholders(specs):
    n_h, n_w, n_c, n_y = specs
    X = tf.compat.v1.placeholder(dtype = tf.float32, shape = [None, n_h, n_w, n_c])
    Y = tf.compat.v1.placeholder(dtype = tf.float32, shape = [None, n_y])
    return X, Y


# In[46]:


def initial_parameters():
    params = {}
    with tf.compat.v1.variable_scope('initial_parameters', reuse = tf.AUTO_REUSE):
        params['W1'] = tf.compat.v1.get_variable('W1', shape = [5, 5, 3, 8], initializer = tf.contrib.layers.xavier_initializer())
        params['W2'] = tf.compat.v1.get_variable('W2', shape = [5, 5, 8, 16], initializer = tf.contrib.layers.xavier_initializer())
    return params


# In[47]:


def forward_propagation(X, params):
    W1, W2 = params['W1'], params['W2']
    #first layer - convolution layer
    C1 = tf.compat.v1.nn.conv2d(X, filters = W1, strides = [1, 1, 1, 1], padding = "VALID")
    A1 = tf.compat.v1.nn.relu(C1)
    #Max Pooling Layer
    P1 = tf.compat.v1.nn.max_pool2d(A1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")
    #second layer - convolution layer
    C2 = tf.compat.v1.nn.conv2d(P1, filters = W2, strides = [1, 1, 1, 1], padding = "VALLID")
    A2 = tf.compat.v1.nn.relu(C2)
    #Max Pooling Layer
    P2 = tf.compat.v1.nn.max_pool2d(A2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")
    #Dropout Layer - 1
    D1 = tf.compat.v1.nn.dropout(P2, rate = 0.25)
    #Dropout Layer - 2
    D2 = tf.compat.v1.nn.dropout(D1, rate = 0.50)
    #Flatten the output layer
    out1 = tf.contrib.layers.fully_connected(D2, num_outputs = 128, activation_fn = tf.compat.v1.nn.relu)
    #Dropout Layer
    out2 = tf.contrib.layers.fully_connected(out1, num_outputs = 1, activation_fn = tf.compat.v1.nn.softmax)
    return out2


# In[48]:


def cost_compute(y_out, Y):
    cost = tf.compat.v1.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits(logits = y_out, labels = Y))
    return cost


# In[49]:


def main(X_train, y_train, learning_rate, epochs):
    costs = []
    m, n_h, n_w, n_c = X_train.shape
    _, n_y = y_train.shape
    X, Y = create_placeholders([n_h, n_w, n_c, n_y])
    parameters = initial_parameters()
    y_out = forward_propagation(X, parameters)
    cost = cost_compute(y_out, Y)
    with tf.compat.v1.variable_scope('main', reuse = tf.AUTO_REUSE):
        optimizer = tf.compat.v1.train.AdamOptimizer(learing_rate = learing_rate).minimize(cost)
        init = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as ses:
            ses.run(init)
            for i in range(epochs):
                _, temp_cost = ses.run([optimizer, cost], feed_dict = {X:X_train, Y:y_train})
                if i%100 == 0:
                    print('the cost after iteration {} is {}'.format(i, temp_cost))
                costs.append(temp_cost)
    #return y_out, parameters, costs


# In[50]:


main(X_train, y_train, 0.06, 800)


# In[ ]:




