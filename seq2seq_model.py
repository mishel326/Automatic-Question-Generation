# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:33:05 2020

@author: mishel
"""

import pickle
import numpy as np 
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding

# Load data (deserialize)
with open('preprocessing.pickle', 'rb') as handle:
    unserialized_data = pickle.load(handle)
    
word2indx = unserialized_data["dictionry"]
embedding_matrix = unserialized_data["matrix_weight"]
indx2word = {v:k for k,v in word2indx.items()}

MAX_LEN_PARAGRAPHS = 200
MAX_LEN_QUESTIONS = len(unserialized_data["train_questions"][0])
VOCAB_SIZE = len(word2indx)
EMBEDDING_DIM = 300

train_paragraphs = np.concatenate((unserialized_data["train_sentence"],unserialized_data["dev_sentence"]),axis = 0)
train_questions = np.concatenate((unserialized_data["train_questions"],unserialized_data["dev_questions"]),axis = 0)
test_paragraphs = unserialized_data["test_sentence"]
test_questions = unserialized_data["test_questions"]
target = train_questions[:,1:]

#para = np.concatenate((train_paragraphs,test_paragraphs),axis = None).reshape(len(para)//200,200)

def embedding_layer_creater(VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN, embedding_matrix):
  
  embedding_layer = Embedding(input_dim = VOCAB_SIZE, 
                              output_dim = EMBEDDING_DIM,
                              input_length = MAX_LEN,
                              weights = [embedding_matrix],
                              trainable = True)
  return embedding_layer

embedding_layer_para = embedding_layer_creater(VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN_PARAGRAPHS, embedding_matrix)
embedding_layer_questions = embedding_layer_creater(VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN_QUESTIONS, embedding_matrix)


def seq2seq_model_builder(HIDDEN_DIM=300):
    
    encoder_inputs = Input(shape=(MAX_LEN_PARAGRAPHS, ),)
    encoder_embedding = embedding_layer_para(encoder_inputs)
    encoder_LSTM = LSTM(HIDDEN_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)
    
    decoder_inputs = Input(shape=(MAX_LEN_QUESTIONS, ),)
    decoder_embedding = embedding_layer_questions(decoder_inputs)
    decoder_LSTM = LSTM(HIDDEN_DIM, return_sequences=True)
    decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=[state_h, state_c])
    
    decoder_dense = Dense(VOCAB_SIZE, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
#    dense_layer = Dense(VOCAB_SIZE, activation='softmax')
##    outputs = TimeDistributed(dense_layer)(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    return model

model = seq2seq_model_builder()
model.summary()
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit([train_paragraphs, train_questions],target, validation_split=0.2,
          batch_size=100,
          epochs=20)