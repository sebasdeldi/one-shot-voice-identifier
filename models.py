
import numpy as np
import random
import sys
import io
import os
import glob

from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D, Flatten
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape, Concatenate,concatenate, Average
from keras.optimizers import Adam
from keras import backend as K
from keras.models import load_model
import tensorflow as tf


def five_average_model(input_shape,base_model):
  model_inputs = []
  model_results = []
  for i in range(5):
    X_input = Input(shape = input_shape, name = "Input"+ str(i))
    model_inputs.append(X_input)
    result = base_model(X_input)
    model_results.append(result)
    
  X = Average()(model_results)
  model = Model(inputs = model_inputs,outputs = X)
  
  return model

def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similariy between u and v
        
    Arguments:
        u -- a word vector of shape (n,)          
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """
    
    distance = 0.0
    
    dot = np.dot(u,v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    cosine_similarity = dot/(norm_u*norm_v)
    
    return cosine_similarity

  
def triplet_loss(y_true, y_pred, alpha = 0.35):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    total_lenght = 64*3
    anchor, positive, negative = y_pred[:,0:int(total_lenght*1/3)],y_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)],y_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)]

    pos_dist = K.sum(K.square(anchor-positive),axis=1)

    neg_dist = K.sum(K.square(anchor-negative),axis=1)

    basic_loss = pos_dist-neg_dist+alpha
    loss = K.sum(K.maximum(basic_loss,0.0))
    
    return loss
  
def base_model(input_shape):
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """
    
    X_input = Input(shape = input_shape)
    
    X = Conv1D(196,kernel_size = 15, strides = 4)(X_input)
    X = Activation('relu')(X)                                
    X = Dropout(rate = 0.2)(X)                                 
    
    X = LSTM(units = 128, return_sequences = True)(X_input)               
    X = Dropout(rate = 0.2)(X)                                 
    
    X = LSTM(units = 128, return_sequences = True)(X)                         
    X = Dropout(rate = 0.2)(X)                                
    
    X = LSTM(units = 128)(X)                                 
    X = Dropout(rate = 0.2)(X)                               
    
    X = Dense(64)(X)
    
    base_model = Model(inputs = X_input, outputs = X)

    return base_model  
  
def speech_model(input_shape, average_model):
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)
    base_model -- model to be used to call the inputs

    Returns:
    model -- Keras model instance
    """
    
    #get triplets vectors
    input_anchor = []
    input_positive = []
    input_negative = []
    for i in range(15):
      X_input = Input(shape = input_shape, name = "Input"+ str(i))
      if (i < 5):
        input_anchor.append(X_input)
      elif (i < 10):
        input_positive.append(X_input)
      elif(i < 15):
        input_negative.append(X_input)
        
    vec_anchor = average_model(input_anchor)
    vec_positive = average_model(input_positive)
    vec_negative = average_model(input_negative)
    
    #Concatenate vectors vec_positive, vec_negative
    concat_layer = concatenate([vec_anchor,vec_positive,vec_negative], axis = -1, name='concat_layer')
    
    model = Model(inputs = input_anchor + input_positive + input_negative, outputs = concat_layer, name = 'speech_to_vec')
    
    
    return model  

def load_model_with_path(model_path):
  return load_model(model_path)

