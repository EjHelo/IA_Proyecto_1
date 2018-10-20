#!/usr/bin/env python
# -*- coding: utf-8 -*-

import normalizacion
import validacion_cruzada

import numpy
from optparse import OptionParser
#from sklearn.preprocessing import LabelEncoder
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.utils import to_categorical

parser = OptionParser()

parser.add_option("", "--arbol", action="store_true", dest="rf", default=False, help="Árbol de decisión")

parser.add_option("", "--cantidad-arbol", dest="cantidad_arbol", default=0,help="Cantidad de arboles")

parser.add_option("", "--umbral-poda", dest="up", default=0, help="Umbral poda")

parser.add_option("", "--red-neuronal", action="store_true", dest="rn", default=False, help="Red neuronal")

parser.add_option("", "--numero-capas", dest="nc", default=0, help="Número de capas")

parser.add_option("", "--unidades-por-capa", dest="uc", default=0,help="Unidades por capa")

parser.add_option("", "--funcion-activacion", dest="fa", default=0,help="Función de activación")

parser.add_option("", "--kfold",action="store_true", dest="kf", default=False, help="KFOLD-CV")

parser.add_option("", "--cantidad-k", dest="cantidad_k", default=0,help="Cantidad de KFolds")

parser.add_option("", "--prefijo", dest="prefijo", default="",help="Prefijo")

parser.add_option("", "--porcentaje-prueba", dest="porcentaje_prueba", default=0,help="Porcentaje de prueba")

(options, args) = parser.parse_args()

#Se les da a los datos el formato necesario dependiendo del tipo de modelo solicitado
if options.rn:
  datos_normalizados = normalizacion.normalizar_rn()
  
  es_entrenamiento = []
  porcentaje_pruebas = int(options.porcentaje_prueba)
  porcion_tamano = (len(datos) * porcentaje_pruebas) // 100
  es_entrenamiento += ['NO'] * porcion_tamano
  es_entrenamiento += ['SI'] * (len(datos) - porcion_tamano)
  es_entrenamiento = numpy.asarray(es_entrenamiento)

  archivo = numpy.concatenate((datos,es_entrenamiento[numpy.newaxis, :].T), axis=1)
 
elif options.rf:
  lista_random, lista_datos_normalizados = normalizacion.normalizar_rf(options.cantidad_arbol)

  vector_archivos =[]
  for x in range( int(options.cantidad_arbol) ):
    es_entrenamiento = []
    porcentaje_pruebas = int(options.porcentaje_prueba)
    porcion_tamano = (len(lista_datos_normalizados[x]) * porcentaje_pruebas) // 100
    es_entrenamiento += ['NO'] * porcion_tamano
    es_entrenamiento += ['SI'] * (len(lista_datos_normalizados[x]) - porcion_tamano)
    es_entrenamiento = numpy.asarray(es_entrenamiento)

    archivo = numpy.concatenate((lista_datos_normalizados[x],es_entrenamiento[numpy.newaxis, :].T), axis=1)
    vector_archivos += [archivo]

atributos = ['id', 'B', 'M', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 
             'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se','smoothness_se', 'compactness_se', 'concavity_se',
             'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 
             'compactness_worst', 'concavity_worst', 'concave points_worst','symmetry_worst', 'fractal_dimension_worst']

#Ejecutamos el CV, se hace kfold
if options.kf == True:
      
      validation_k = int(options.cantidad_k)
      cantidad_arboles = int(options.cantidad_arbol)
      

      if options.rf:
        #se crean los arboles
        for x in range(cantidad_arboles):
          respuestas, fold_error_t, fold_error_v, final_error_t, final_error_v = validacion_cruzada.k_fold_cross_validation(validation_k, porcentaje_pruebas, lista_datos_normalizados[x], lista_random[x], options)
          print("Información del Árbol número: ", x)
          print("fold_accuracy_T", 100 - fold_error_t)
          print("fold_accuracy_V", 100 - fold_error_v)
          print("final_accuracy_T",100 - final_error_t)
          print("final_accuracy_V",100 - final_error_v)
          respuestas = numpy.asarray(respuestas)

          vector_archivos[x] = numpy.concatenate((archivo,respuestas[numpy.newaxis, :].T), axis=1)
          #-------------------------------------
          vector_archivos[x] = (vector_archivos[x]).tolist()
          archivo_final = []
          archivo_final.append(atributos) 
          archivo_final += vector_archivos[x]

          nombre_archivo = options.prefijo + ".csv"
          #createCSV(nombre_archivo,archivo_final)
        
      else:
        #Se aplica cross-validation
        respuestas, fold_error_t, fold_error_v, final_error_t, final_error_v = validacion_cruzada.k_fold_cross_validation(validation_k, porcentaje_pruebas, datos_normalizados, [], options)
        print("fold_accuracy_T", 100 - fold_error_t)
        print("fold_accuracy_V", 100 - fold_error_v)
        print("final_accuracy_T",100 - final_error_t)
        print("final_accuracy_V",100 - final_error_v)
        respuestas = numpy.asarray(respuestas)

        archivo = numpy.concatenate((archivo,respuestas[numpy.newaxis, :].T), axis=1)
        #--------------------------------
        archivo = archivo.tolist()
        archivo_final = []
        archivo_final.append(atributos) 
        archivo_final+= archivo

        nombre_archivo = options.prefijo + ".csv"
        #createCSV(nombre_archivo,archivo_final)



#--arbol --cantidad-arbol 4 --umbral-poda 0.1 --kfold --cantidad-k 5 -prefijo 'prueba'
#--porcentaje-pruebas 25 



