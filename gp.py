
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

