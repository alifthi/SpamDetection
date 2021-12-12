# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 17:14:46 2021

@author: Ali Fthi
"""
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow.keras.layers as ksl
import numpy as np
import tensorflow as tf
import pandas as pd

print('using tensorflow version 2.7.0')
Data = pd.read_csv('spam_or_not_spam.csv')



print('******************\n',Data.label.value_counts(),'\n******************')

Target = Data['label']
Target = tf.keras.utils.to_categorical(Target,num_classes=2)

Data['PPText'] = Data['email'].str.replace('[^\w\s]','')
Data['PPText'] = Data['PPText'].fillna('')
Data['PPText'] = Data['PPText'].str.lower()

MaxWords = 5000
Tknizr = tf.keras.preprocessing.text.Tokenizer(num_words=MaxWords)

Tknizr.fit_on_texts(list(Data['PPText']))
Data['Seq'] = Tknizr.texts_to_sequences(list(Data['PPText']))

VSize = len(Tknizr.word_index) + 1 

Data['Size'] = Data['PPText'].apply(lambda x : len(x))
print('Describe length of Data:\n',Data['Size'].describe())

MaxLen = 2000
TrainData = tf.keras.preprocessing.sequence.pad_sequences(Data['Seq'],
                                                            maxlen = MaxLen)

xTrain,xTest,yTrain,yTest = train_test_split(TrainData,Target,test_size=0.2)

EmbeddigDim = 100



model = tf.keras.models.Sequential()

model.add(ksl.Embedding(input_dim = VSize,output_dim = EmbeddigDim,
                            input_length = MaxLen))

model.add(ksl.Flatten())
model.add(ksl.Dense(2,activation = 'sigmoid'))



model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['accuracy'])


print('******************\nmodel Summary:\n')
model.summary()


Hist = model.fit(xTrain,yTrain,validation_data=(xTest,yTest),batch_size=30,
                 epochs=5)


plt.plot(Hist.epoch,Hist.history['loss'])
plt.plot(Hist.epoch,Hist.history['val_loss'])
plt.legend(['TrainLoss','ValidationLoss'])
plt.show()
plt.plot(Hist.epoch,Hist.history['accuracy'])
plt.plot(Hist.epoch,Hist.history['val_accuracy'])
plt.legend(['TrainAccuracy','ValidationAccuracy'])


model.save('Model')

res = np.round(model.predict(xTest))
res = [np.argmax(x) for x in res]
yTest = [np.argmax(x) for x in yTest]
CM = confusion_matrix(yTest,res)
print('******************************\nConfusion Matrix:\n',CM)
