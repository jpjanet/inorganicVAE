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
m = np.shape(data)[0]
d = np.shape(data)[1]