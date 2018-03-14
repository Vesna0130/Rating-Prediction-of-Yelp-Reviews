#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
import keras.preprocessing.sequence as S
from keras.utils import to_categorical
from keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
import pandas as pd
#import json
import numpy as np
from nltk.tokenize import word_tokenize

vocab_size = 5000
sentence_max_len = 400
model_path = 'keras.h5'

#os.chdir('/Users/vesna/VesnaLi/Courses/STAT628_Data_Science_Practicum/Module2')

        
class SentimentLSTM:
    def __init__(self):
        self.tokenizer = Tokenizer(num_words=vocab_size)
        self.stop_words = []
        self.model = None

    def load_stop_word(self,path='data/new_stop.txt'):
        with open(path, 'r') as f:
            for line in f:
                content = line.strip()
                self.stop_words.append(content.decode('utf-8'))

    def jieba_cut(self,line):
        lcut = word_tokenize('line')
        cut = [x for x in lcut if x not in self.stop_words]
        cut = " ".join(cut)
        return cut

    def load_cuted_corpus(self, dir, input):
        """
        f = open(dir + '/' + input , 'r')
        lines = f.readlines()
        texts = []
        labels = []
        for line in lines:
            fields = line.split(',')
            rate = int(fields[0])
            cont = fields[1:]
            cont = " ".join(cont)
            texts.append(cont)
            labels.append((rate-1))

        self.tokenizer.fit_on_texts(texts)
        f.close()
        """
        train = pd.read_csv(dir + '/' + input)
        train = train.sample(150000)
        texts = train['trans_text'].tolist()
        labels = train['stars'].tolist()
        
        self.tokenizer.fit_on_texts(texts)
        return texts,labels

    def load_data(self):
        x_train,y_train = self.load_cuted_corpus('data', 'train_trans.csv')
        x_train = self.tokenizer.texts_to_sequences(x_train)
        x_train = S.pad_sequences(x_train,maxlen=sentence_max_len)
        y_train = to_categorical(y_train)
        
        return ((x_train[0:100000],y_train[0:100000]), (x_train[100000:], y_train[100000:]))

    def train(self,epochs=50):
        print('building model ...')
        self.model = SentimentLSTM.build_model()

        print('loading data ...')
        (text_train, rate_train), (text_test, rate_test) = self.load_data()

        print('training model ...')
        self.model.fit(text_train, rate_train,batch_size=1000,epochs=epochs)
        self.model.save('model/keras.model')
        score = self.model.evaluate(text_test,rate_test)
        print(score)

    def load_trained_model(self,path):
        model = SentimentLSTM.build_model()
        model.load_weights(path)
        return model

    def predict_text(self,text):
        if self.model == None:
            self.model = self.load_trained_model(model_path)
            self.load_stop_word()
            self.load_cuted_corpus('data', 'LSTM_train.csv')

        vect = self.jieba_cut(text)
        vect = vect.encode('utf-8')
        vect = self.tokenizer.texts_to_sequences([vect,])
        print(vect)
        return self.model.predict_classes(S.pad_sequences(np.array(vect),sentence_max_len))

    @staticmethod
    def build_model():
        model = Sequential()
        model.add(Embedding(vocab_size, 100, input_length=sentence_max_len))
        model.add(Bidirectional(LSTM(128,implementation=2)))
        model.add(Dropout(0.2))
        model.add(Dense(6, activation='softmax'))
        model.compile('RMSprop', 'categorical_crossentropy', metrics=['accuracy'])
        return model


#if __name__=="__main__":
    lstm = SentimentLSTM()
    lstm.train(6)
    while True:
        Input = input('Please input text:')
        if Input == 'quit':
            break
        print(lstm.predict_text(input))