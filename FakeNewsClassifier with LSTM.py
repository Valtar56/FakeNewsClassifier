# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 12:53:28 2021

@author: SUDIP WALTER THOMAS
"""
import pandas as pd

df = pd.read_csv('./train.csv/train.csv')

# drop na values
df=df.dropna()
# get independent features
X = df.drop('label',axis = 1)
# dependent features
y = df['label']

import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences # to make the sentences equal in length
from tensorflow.keras.preprocessing.text import one_hot # converting a sentence into one hot representation given a vocabulary size
from tensorflow.keras.models import Sequential
# final layer gives probabilty, if greater than 0.5 then fake otherwise not fake
# specifying vocabulary size
voc_size = 5000

# One Hot Representation
text = X.copy()
text['title'][0]
# because we dropped nan values, so we need to reset index
text.reset_index(inplace=True)

# Data PreProcessing
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
corpus = []
for i in range(0, len(text)):
    review = re.sub('[^a-zA-Z]', ' ',text['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


onehot_repr=[one_hot(words,voc_size)for words in corpus]
# Embedding Representation
sent_length=20 # the sentences will get padded and be made each of length 20
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)

# Creating the architecture of the model
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)

# Model Training
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)

# Adding dropout to the same model and testing accuracy
from tensorflow.keras.layers import Dropout
model_ = Sequential()
model_.add(Embedding(voc_size, embedding_vector_features, input_length = sent_length))
model_.add(Dropout(0.3))
model_.add(LSTM(100))
model_.add(Dropout(0.3))
model_.add(Dense(1, activation = 'sigmoid'))
model_.compile(loss = 'binary_crossentropy', optimizer = 'adam',metrics=['accuracy'])

model_.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)
# Performance & Accuracy
# without droppout
y_pred = model.predict_classes(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm= confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)

# with dropout

y_pred_ = model_.predict_classes(X_test)
cm_= confusion_matrix(y_test,y_pred_)
acc_ = accuracy_score(y_test,y_pred_)









