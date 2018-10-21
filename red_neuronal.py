import tensorflow as tf
from tensorflow import keras
import keras
from keras.models import Sequential
from keras.layers import Dense
#import normalizacion
import pandas as pd


def crear_red_neuronal(training_set, validation_set, capas, unidades_por_capa, pred=0):

    training_set_x = training_set.drop(columns=['B','M'])
    training_set_y = training_set.iloc[:,[1,2]]
    validation_set_x = validation_set.drop(columns=['B','M'])
    validation_set_y = validation_set.iloc[:,[1,2]]
    
    model = Sequential()
    model.add(Dense(unidades_por_capa, input_shape=(31,), activation = "sigmoid"))
    for i in range (0,capas):
        model.add(Dense(unidades_por_capa, input_shape=(31,), activation = "sigmoid"))
        
    model.add(Dense(2, activation = "sigmoid"))
    
    tbCallBack = keras.callbacks.TensorBoard(log_dir='/tmp/keras_logs',write_graph = True)

    model.compile(loss='mean_squared_logarithmic_error', optimizer="sgd", metrics= ['accuracy'])
 

    model.fit(training_set_x, training_set_y ,batch_size=100 , epochs= 100, verbose= 0)

    scores = model.evaluate(training_set_x, training_set_y)
    scores2 = model.evaluate(validation_set_x, validation_set_y)

    if(pred==0):
        return scores, scores2
    else:
        validation_pred = model.predict_classes(validation_set_x, batch_size = 100)
        training_pred = model.predict_classes(training_set_x, batch_size = 100)
        return  training_pred, validation_pred

