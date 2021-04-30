#imports
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense , Embedding ,GlobalAveragePooling1D
from tensorflow.python.keras.callbacks import LambdaCallback
import matplotlib.pyplot as plt
import pickle



(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words = 10000)
print(x_train[0])
class_names = ['Negative', 'Positive']
word_index = imdb.get_word_index()
print(word_index['hello'])
reverse_word_index = dict((value,key) for key,value in word_index.items())

def decode(review):
    text = ''
    for i in review:
        text+=reverse_word_index[i]
        text+=" "
        
    return text

decode(x_train[0])


def show_len():
    print("length of the 1st training example :",len(x_train[0]))
    print("length of the 2nd training example :",len(x_train[1]))
    print("length of the 1st test example :",len(x_test[0]))
    print("length of the 1nd test example :",len(x_test[1]))
    
show_len()

word_index['the']

x_train = pad_sequences(x_train, value = word_index['the'], padding='post', maxlen=256)
x_test = pad_sequences(x_test, value = word_index['the'], padding='post', maxlen=256)

show_len()
decode(x_train[0])

#creating the model
model = Sequential([
    Embedding(10000, 16),
    GlobalAveragePooling1D(),
    Dense(16 , activation = 'relu'),
    Dense(1 , activation = 'sigmoid')
])

#compiling the model

model.compile(loss = 'binary_crossentropy',
             optimizer = 'adam',
             metrics = ["accuracy"])
model.summary()

simple_log = LambdaCallback(on_epoch_end = lambda e,l : print(e, end='.'))

E = 20
h = model.fit(x_train,
                 y_train,
                 validation_split = 0.2,
                 epochs = E,
                 callbacks = [simple_log],
                 verbose = False
                 )
h


loss, acc = model.evaluate(x_test,y_test)
print("The loss:", loss)
print("The accuracy of the test set:",acc*100)

import numpy as np
p = model.predict(np.expand_dims(x_test[0], axis = 0))
print(class_names[int(p[0]>0.5)])



