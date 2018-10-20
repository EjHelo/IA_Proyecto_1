import tensorflow as tf
from tensorflow import keras
import keras
from keras.models import Sequential
from keras.layers import Dense

def red_neuronal(capas, dataset, atributos):
    print(dataframe_normalizado.shape[0])
    model = Sequential()
    model.add(Dense(len(atributos[0]), input_dim=dataframe_normalizado.shape[0], activation = "sigmoid"))
    for i in capas:
        model.add(Dense(33, input_dim=dataframe_normalizado.shape[0], activation = "sigmoid"))
        
    #model.summary()
    model.add(Dense(len(atributos[0]), activation = "sigmoid"))
    
    tbCallBack = keras.callbacks.TensorBoard(log_dir='/tmp/keras_logs',write_graph = True)

    model.compile(loss='categorical_crossentropy', optimizer="sgd", metrics= ['accuracy'], target_tensors = 'B')
    print(dataframe_normalizado.keys())

    model.fit(dataset, dataset.keys(), epochs= 10, verbose= 2, callback = [tbCallBack])

