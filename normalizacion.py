
import pandas as pd
import numpy as np
import random

def normalizar_rf(cantidad):
  cantidad_arbol = int (cantidad)
  #leemos el CSV
  dataframe = pd.read_csv("wisc_bc_data.csv",parse_dates=True)

  vector_atributos = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 
                    'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se','smoothness_se', 'compactness_se', 'concavity_se',
                    'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 
                    'compactness_worst', 'concavity_worst', 'concave points_worst','symmetry_worst', 'fractal_dimension_worst']


  dataframe_normalizado = normalizar_dataframe(dataframe, vector_atributos)

  muestras = np.transpose(dataframe_normalizado) #generamos variabe muestra a trabajar

  #vector en donde estan todas las salidas_temporales de los arboles
  vector_general = [] 

  #asignamos la cantidad de atributos segun la cantidad de arboles
  if cantidad_arbol > 1 and (cantidad_arbol) <= 4:
      cantidad_atributos = 15
      
  if cantidad_arbol > 4 and cantidad_arbol <= 8:
      cantidad_atributos = 10
      
  if cantidad_arbol > 8:
      cantidad_atributos = 5
      
      
  random_general=[]
  for _ in range(cantidad_arbol):
    #Hacemos el random del arbol
    vector_random = [random.randint(3,31) for _ in range(cantidad_atributos)]
    print('VecRan',vector_random)
    #Guardamos todos los vectores random generados
    random_general += [vector_random]
    
    #---------------------------------------
    arbol_set= dataframe.iloc[:,vector_random]
    arbol_set.to_csv('arbolSet.csv', sep=',')
    #-------------------------------------
      
    #Ingresamos los tres generales
    identificador = (muestras[0])[np.newaxis, :].T
    benigno = (muestras[1])[np.newaxis, :].T
    maligno = (muestras[2])[np.newaxis, :].T
    salida_temporal = np.concatenate((identificador, benigno, maligno), axis = 1)

    for var in range (len(vector_random)):
      new_var = (muestras[ (vector_random[var]) ])[np.newaxis, :].T
      salida = np.concatenate((salida_temporal, new_var ), axis = 1)
      salida_temporal = salida

    vector_general += [ (salida_temporal.tolist()) ]
  print('vecGen', len(vector_general))
  return random_general, vector_general



def normalizar_dataframe(dataframe, vector_atributos):
  #Hacemos dummie de la variable diagnosis
  dummies_diagnosis = pd.get_dummies(dataframe.diagnosis)

  #Adjuntamos los dummies y borramos la anterior
  dataframe_temporal = pd.concat([dataframe, dummies_diagnosis], axis='columns')
  dataframe_normalizado = dataframe_temporal.drop(['diagnosis'], axis='columns')

  #Hacemos normalización Z-score
  for var in range (len(vector_atributos)):
    old_var = vector_atributos[var]
    new_var = 'z_'+ old_var
    dataframe_normalizado[new_var] = round( (dataframe[old_var] - dataframe[old_var].mean()) / dataframe[old_var].std(ddof=0) )
  #eliminamos las variables anteriores
  dataframe_normalizado = dataframe_normalizado.drop(vector_atributos, axis='columns')

  diccionario = {-6.0:'<-2', -5.0:'<-2', -4.0:'<-2', -3.0:'<-2', -2.0:'<-2', 
               -1.0:'-1 _ 1', 0.0:'-1 _ 1', 1.0:'-1 _ 1', 
               2.0:'2 _ 3', 3.0:'2 _ 3', 
               4.0:'4+', 5.0:'4+', 6.0:'4+', 7.0:'4+', 8.0:'4+'}

  for x in range(len(vector_atributos)):
    dataframe_normalizado.replace({ ('z_'+vector_atributos[x]): diccionario }, inplace = True)

  return dataframe_normalizado



