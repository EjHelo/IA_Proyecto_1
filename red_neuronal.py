import tensorflow as tf
from tensorflow import keras
import keras
from keras.models import Sequential
from keras.layers import Dense
#import normalizacion
import pandas as pd

def red_neuronal(capas, dataset, atributos):
    '''
    datos_normalizados = normalizacion.normalizar_rn()
    #print(dataframe_normalizado.shape[0])
    datos1 = datos_normalizados.iloc[:33,:]
    datos2 = datos_normalizados.iloc[33:,:]
    print(datos_normalizados.iloc[:,2:])
    datos3 = datos_normalizados.drop(columns=['B','M'])
    
    print(datos_normalizados.shape[0])
    model = Sequential()
    model.add(Dense(dataset, input_shape=(31,), activation = "sigmoid"))
    for i in range (0,capas):
        model.add(Dense(dataset, input_shape=(31,), activation = "sigmoid"))
        
    #model.summary()
    model.add(Dense(2, activation = "linear"))
    
    tbCallBack = keras.callbacks.TensorBoard(log_dir='/tmp/keras_logs',write_graph = True)

    model.compile(loss='mean_squared_logarithmic_error', optimizer="sgd", metrics= ['accuracy'])
    #print(dataframe_normalizados.keys())

    model.fit(datos3, datos_normalizados.iloc[:,[1,2]],batch_size=100 , epochs= 100, verbose= 0)
    porcion = datos_normalizados.shape[0]//5
    datos4 = datos_normalizados.iloc[porcion: (porcion+porcion), :]
    datos5 = datos_normalizados.iloc[0:porcion, :]
    datos6 = datos_normalizados.iloc[porcion+porcion: datos_normalizados.shape[0], :]
    datos7 = [datos5,datos6]
    datos8 = pd.concat(datos7)
    print("LEN _---------- ")
    print (len(datos8))
    print(datos8)
    #score = model.evaluate(datos_normalizados, datos_normalizados.iloc[:,1])
    #score = model.predict(datos3, verbose=1)
    #score = model.evaluate(datos3.iloc[150:,:], datos_normalizados.iloc[150:,[1,2]], verbose=1)
    #print(score)
   
    score = model.predict_classes(datos3, batch_size=50)
    predicciones = []

    for i in range(len(datos3)):
        #print("Predicted=%s" % (yPredict[i]))
        predicciones.append(score[i])
                    
    print(predicciones)
    '''
def crear_red_neuronal(training_set, validation_set, capas, unidades_por_capa):

    training_set_x = training_set.drop(columns=['B','M'])
    training_set_y = training_set.iloc[:,[1,2]]
    validation_set_x = validation_set.drop(columns=['B','M'])
    validation_set_y = validation_set.iloc[:,[1,2]]
    #print("DATOS ===============================================================/n")
   # print(validation_set_x)
    
    model = Sequential()
    model.add(Dense(unidades_por_capa, input_shape=(31,), activation = "sigmoid"))
    for i in range (0,capas):
        model.add(Dense(unidades_por_capa, input_shape=(31,), activation = "sigmoid"))
        
    #model.summary()
    model.add(Dense(2, activation = "sigmoid"))
    
    tbCallBack = keras.callbacks.TensorBoard(log_dir='/tmp/keras_logs',write_graph = True)

    model.compile(loss='mean_squared_logarithmic_error', optimizer="sgd", metrics= ['accuracy'])
    #print(dataframe_normalizados.keys())

    model.fit(training_set_x, training_set_y, batch_size=100 , epochs= 100, verbose= 0, validation_split = 0.3)

    scores = model.evaluate(training_set_x, training_set_y)
    scores2 = model.evaluate(validation_set_x, validation_set_y)

    print(scores)
    print('/n')
    print(scores2)

    pred = model.predict_classes(validation_set_x, batch_size = 100);
    print("prediccionbes: ====================================/n ")
    print(pred)

    return scores[1], scores2[1]

