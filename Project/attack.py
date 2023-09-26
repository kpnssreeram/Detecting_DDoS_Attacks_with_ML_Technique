import arff 
import numpy as np
from matplotlib import pyplot as plt 
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras.utils import to_categorical

file=open("final-dataset.arff", 'r')


def scrape_data():
    decoder=arff.ArffDecoder()
    data=decoder.decode(file, encode_nominal=True)
    vals=[val[0: -1] for val in data['data']]
    labels =[label[-1] for label in data['data']]

    for val in labels:
        if labels[val] != 0:
            labels[val]=1
    
    training_data=vals[0: int(.9*len(vals))]
    training_labels=labels[0: int(.9*len(vals))]
    validation_data=vals[int(.9*len(vals)):]
    validation_labels=labels [int(.9*len(vals)):]

    training_labels=to_categorical(training_labels, 5)
    validation_labels=to_categorical(validation_labels, 5)
    
    return np.asarray(training_data), np.asarray(training_labels), np.asarray(validation_data),np.asarray(validation_labels)


def generate_model(shape): 
    model=Sequential()
    model.add(Dense (30, input_dim=shape, kernel_initializer="uniform", activation='relu'))

    model.add(Dropout (0.4)) 
    model.add(Dense (10, activation="relu"))
    model.add(Dropout (0.4)) 
    model.add(Dense (10, activation='relu'))

    model.add(Dropout (0.4)) 
    model.add(Dense (64, activation='relu'))
    

    model.add(Dropout (0.4))
    model.add(Dense(5, activation='softmax'))
    print (model.summary())
    
    return model


data_train,label_train,data_eval, label_eval =scrape_data()

model=generate_model(len(data_train[0]))

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])


tensorboard=TensorBoard(log_dir='logs/', histogram_freq=0, write_graph=True, write_images=True)


history=model.fit(data_train, label_train, validation_data=( data_eval, label_eval), epochs=2, callbacks=[tensorboard])

loss_history=history.history["loss"]


print(model.evaluate(data_eval, label_eval))
print(model.evaluate(data_train, label_train))