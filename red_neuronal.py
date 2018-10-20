import tensorflow as tf
from tensorflow import keras
import keras
from keras.models import Sequential
from keras.layers import Dense


print(dataframe_normalizado.shape[0])
model = Sequential()
model.add(Dense(33, input_dim=dataframe_normalizado.shape[0], activation = "sigmoid"))
model.add(Dense(33, input_dim=dataframe_normalizado.shape[0], activation = "sigmoid"))
model.add(Dense(33, input_dim=dataframe_normalizado.shape[0], activation = "sigmoid"))
model.add(Dense(len(dataframe_normalizado.keys()), activation = "sigmoid"))
#model.summary()

tbCallBack = keras.callbacks.TensorBoard(log_dir='/tmp/keras_logs',write_graph = True)

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics= ['accuracy'])
print(dataframe_normalizado.keys())

model.fit(dataframe_normalizado, dataframe_normalizado.keys(), epochs= 10, verbose= 2, callback = [tbCallBack])
