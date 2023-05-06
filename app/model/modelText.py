# -*- coding: utf-8 -*-
"""
Created on Sat May  6 03:15:00 2023

@author: Le Thai Nhat Nam
"""
import tensorflow as tf

import pickle
with open('model/tokenizer1.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
from keras.utils import pad_sequences
new_model = tf.keras.models.load_model('./model/my_model')

def predict_pickle(text):
    test_sen = [text]
    test_seq = tokenizer.texts_to_sequences(test_sen)
    print(test_seq)
    padded_test_seq = pad_sequences(test_seq, maxlen=140, truncating="post", padding="post")
    return new_model.predict(padded_test_seq)[0][0]
