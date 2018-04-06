
## based on 
# https://danijar.com/building-variational-auto-encoders-in-tensorflow/
# https://towardsdatascience.com/teaching-a-variational-autoencoder-vae-to-draw-mnist-characters-978675c95776
## Imports

import numpy as np
import tensorflow as tf
import pandas as pd
import sklearn.model_selection
import os, glob, shutil, math, time
from nnrunfunction import  *
# reset tf graph
tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.ERROR)

## data pipeline from R-prepared CSV

df = pd.read_csv("split_x_vae.csv",sep=",")
data = df.values

m = np.shape(data)[0]
d = np.shape(data)[1]

## Functions
def data_rescale(scaled_dat,train_mean,train_var):
    d = np.shape(train_mean)[0]
    print('unnormalizing with number of dimensions = ' +str(d))
    dat = (np.multiply(scaled_dat.T,np.sqrt(train_var),)+train_mean).T
    return(dat)
def data_normalize(data,train_mean,train_var):
    d = np.shape(train_mean)[0]
    print('normalizing with number of dimensions = ' +str(d))
    scaled_dat = np.divide((data.T - train_mean),np.sqrt(train_var),).T
    return(scaled_dat)
    
## data generation
train_data_x = data[msk]


# non-dimenisionalization
train_var_x = np.var(train_data_x,0).reshape(d,1)
train_mean_x = np.mean(train_data_x,0).reshape(d,1)
scaled_train_data_x = data_normalize(train_data_x,train_mean_x,train_var_x)


## tf variable placeholders for feeding
with tf.name_scope('input'):
    X = tf.placeholder(tf.float32,[None,n_input],name="X")
    Y = tf.placeholder(tf.float32,[None,n_input],name="Y")
    global_step = tf.Variable(0, trainable=False)
    keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')



def encoder(features mode, params):
    """ NN estimator function """
    layers = params["layers"]
    topology = params["topology"]
    L = []
    p=  params["latent_dim"]
    # set up regularizer
    regularizer = tf.contrib.layers.l2_regularizer(scale=params["l2_lambda"])
    # input layer will just be features
    L.append(features)
    for l in range(1,layers+1):
        # Hidden fully connected layer with n_hidden_1 neurons
        L.append(tf.layers.dense(inputs = L[l-1],units= topology[l-1], activation = tf.nn.relu, kernel_regularizer=regularizer, name=str('layer_'+str(l+1))))
    print(L)
    print('length of layers ' + str(len(L)))
    # Output fully connected layer with p neurons
    output_layer = tf.layers.dense(inputs = L[layers],units= p, activation= None,kernel_regularizer=regularizer,name='output_layer')
    # Reshape output layer to 1-dim Tensor to return predictions
    predictions = tf.reshape(output_layer, [-1])
    
        activation = lrelu
    with tf.variable_scope("encoder", reuse=None):
        X = tf.reshape(X_in, shape=[-1, 28, 28, 1])
        x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        mn = tf.layers.dense(x, units=n_latent)
        sd       = 0.5 * tf.layers.dense(x, units=n_latent)            
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent])) 
        z  = mn + tf.multiply(epsilon, tf.exp(sd))
        
        return z, mn, sd
    return predictions
