'''
@author: Michael Guarino
desc: this file contains all lstm functions
notes:
'''

from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM
from keras.optimizers import Adam
import csv

import numpy as np

from utils import SEQUENCE_LENGTH, EPOCHS

def train_model(trainData, trainLabel):
    '''
    desc: runs the lstm which preforms the sequence to sequence modeling task
          and saves the model
    args: a one hot encoded 3 dimensional matrix of training data and
          testing data, which represents the next element in the
          sequence
    '''
    #set number of unique words in vocabulary
    unq_words = trainData.shape[2]

    #lstm model
    model = Sequential()
    model.add(LSTM(unq_words, dropout_W=0.2, dropout_U=0.2, return_sequences=True, input_dim=unq_words))
    model.add(Dense(unq_words))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(trainData, trainLabel, batch_size=1, epochs=EPOCHS)
    model_json = model.to_json()
    with open("lstm_model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("lstm_model.h5")
#end

def load_model():
    '''
    desc: loads lstm model
    args: returns loaded model object
    '''
    json_file = open('lstm_model.json', 'r')
    loaded_lstm_model_json = json_file.read()
    json_file.close()
    loaded_lstm_model = model_from_json(loaded_lstm_model_json)
    loaded_lstm_model.load_weights("lstm_model.h5")
    loaded_lstm_model.compile(loss='binary_crossentropy', optimizer='adam')
    return loaded_lstm_model
#end

def test_model(model, testData, unqVoc_LookUp, inputString):
    '''
    desc: runs a prediction task on a given sequence
    args: one hot encoded padded sequence
    returns: prediction of the next word in the sequence
    '''
    nextTimeStep = len(inputString.split(' ')) - 1
    pred = model.predict(testData, verbose=0)
    wordDist = np.argmax(pred, axis=2)
    prob = np.max(pred, axis = 2)[0][nextTimeStep]
    word = list(unqVoc_LookUp.keys())[list(unqVoc_LookUp.values()).index(wordDist[0][nextTimeStep])]
    print("Next word is : {}, with a proabability of {}".format(word, prob))
#end
