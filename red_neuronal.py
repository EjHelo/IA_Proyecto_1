import tensorflow as tf
from tensorflow import keras
import keras
from keras.models import Sequential
from keras.layers import Dense
import normalizacion

def red_neuronal(capas, dataset, atributos):
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

    model.fit(datos3, datos_normalizados.iloc[:,[1,2]],batch_size=100 , epochs= 100, verbose= 2)

    #score = model.evaluate(datos_normalizados, datos_normalizados.iloc[:,1])
    #score = model.predict(datos3, verbose=1)
    score = model.evaluate(datos3.iloc[150:,:], datos_normalizados.iloc[150:,[1,2]], verbose=1)
    print(score)
    

