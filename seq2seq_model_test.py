# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 19:05:59 2020

@author: mishel
"""
import pickle
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# Load data (deserialize)
with open('preprocessing_keras.pickle', 'rb') as handle:
    unserialized_data = pickle.load(handle)
    
x_tr = unserialized_data["train_paragraphs"]
y_tr = unserialized_data["train_questions"]
x_val = unserialized_data["dev_paragraphs"]
y_val = unserialized_data["dev_questions"]
x_voc_size = unserialized_data["x_voc_size"]
y_voc_size = unserialized_data["y_voc_size"]
max_len_para = unserialized_data["max_len_para"]
max_len_ques = unserialized_data["max_len_ques"] - 1
latent_dim = 300

# Encoder 
encoder_inputs = Input(shape=(max_len_para,)) 
enc_emb = Embedding(x_voc_size, latent_dim,trainable=True)(encoder_inputs)

#LSTM Encoder
encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True) 
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb) 

# Set up the decoder. 
decoder_inputs = Input(shape=(max_len_ques,)) 
dec_emb_layer = Embedding(y_voc_size, latent_dim,trainable=True) 
dec_emb = dec_emb_layer(decoder_inputs)  

#LSTM using encoder_states as initial state
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True) 
decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h1, state_c1]) 

#Dense layer
decoder_dense = Dense(y_voc_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs) 

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs) 
model.summary()
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
history=model.fit([x_tr,y_tr[:,:-1]], y_tr[:,1:] ,epochs=2,callbacks=[es],batch_size=512, validation_data=([x_val,y_val[:,:-1]], y_val[:,1:]))