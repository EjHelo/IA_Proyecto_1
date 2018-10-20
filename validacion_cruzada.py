import numpy
import copy

import arbol_decision
#import redes_neuronales

def obtener_error_rate(resultados, resultados_reales):
    errores = 0
    for indice_resultado in range(len(resultados)):
        if (resultados[indice_resultado] != resultados_reales[indice_resultado]):
            errores += 1

    return errores

#Recibe un set de entrenamiento y retorna una lista con las respuestas esperadas por cada ejemplo
def obtener_resultados_reales(matrix):
    return [row[len(row)-1] for row in matrix]

#Recibe un training set y un validation set. Busca el tipo de learner que se va a usar en options, asi como otros parametros necesarios
def obtener_resultados(training_set, validation_set, vector_random, modelo):
    result_training = []
    result_validation = []

    if modelo.rn == True:
        print("Realizando redes neuronales")

  
    elif modelo.rf == True:
    
        #Obtenemos los atributos y el target, q van a variar dependiendo del tipo de corrida
        atributos = vector_random
        target = 'B'
        #print('atributos', atributos)

        #Generamos el arbol
        arbol = arbol_decision.crear_arbol(training_set, atributos, target)
    
        #Realizamos la poda
        arbol_decision.podar_arbol(arbol, float(modelo.up))
   
    
        #Realizamos las predicciones con el training set
        for example in training_set:
            diccionario_example = {}
            for i in range(len(example)):
                diccionario_example[atributos[i]] = example[i]

            nuevo_resultado = arbol_decision.prediccion_arbol_decision(arbol, diccionario_example)
            result_training.append(nuevo_resultado)

        #Realizamos las predicciones con el validation set
        for example in validation_set:
            diccionario_example = {}
            for i in range(len(example)):
                diccionario_example[atributos[i]] = example[i]
            nuevo_resultado = arbol_decision.prediccion_arbol_decision(arbol, diccionario_example)
            result_validation.append(nuevo_resultado)

    return result_training, result_validation

#Retorna el training y validation set para un kfold cv
def particion_k(examples, i, validation_k):
    training_set = []
    validation_set = []
        
    tamano_porcion = len(examples)//validation_k
    training_set += examples[0: i*tamano_porcion] #Primera parte del training_set
    validation_set += examples[i*tamano_porcion:i*tamano_porcion+tamano_porcion] #Validacion set
    training_set += examples[i*tamano_porcion+tamano_porcion:len(examples)] #Todo lo que sobra va para el training

    return training_set, validation_set 


#Retorna el training y validation set para un hold out cv
def particion_h(examples, porcentaje_prueba):

    tamano_porcion = (len(examples) * porcentaje_prueba) // 100
    validation_set = numpy.concatenate( [ ], examples.iloc(:tamano_porcion,:))
    training_set = numpy.concatenate( [ ], examples.iloc(tamano_porcion:,:))
    data = examples.iloc[:tamano_porcion,:]
    data2 = examples.iloc[tamano_porcion:,:]
    print(data)
    print(data2)
    #validation_set += examples[0:tamano_porcion] #Validacion set
    
    #training_set += examples[tamano_porcion:len(examples)] #Todo lo que sobra va para el training

    return training_set, validation_set

#Recibe un tipo de learner, retorna el error promedio usando el training set, y el error promedio usando el validation
def k_fold_cross_validation(k_validaciones, porcentaje_pruebas, examples, vector_random, modelo):
    fold_error_t = 0
    fold_error_v = 0

    #-------------------------------------------------------
    #Dejamos un 70% para fold, y un 30% para test set
    k_fold_examples, test_set = particion_h(examples, porcentaje_pruebas)
    #Hacemos copias de los valores
    k_validaciones_original = k_validaciones
    k_fold_examples_original = numpy.copy(k_fold_examples)
    test_set_original = numpy.copy(test_set)

    if len(k_fold_examples) % k_validaciones != 0:
        k_validaciones +=1
    #-------------------------------------------------------

    for i in range(k_validaciones):
        #Entrenamiento y validacion con particion K
        training_set, validation_set = particion_k(k_fold_examples, i, k_validaciones_original)
                
        #Hacemos copias
        training_set_original = numpy.copy(training_set)
        validation_set_original = numpy.copy(validation_set)
           
        result_training, result_validation = obtener_resultados(training_set, validation_set, vector_random, modelo)

        #Si es una red (rn) entonces ya en result_training y result_validation tengo el error rate
        if modelo.rn:
            fold_error_t += result_training
            fold_error_v += result_validation

        #Si no, hay que calcularlo   
        else:
            fold_error_t += obtener_error_rate(result_training, obtener_resultados_reales(training_set_original) )
            fold_error_v += obtener_error_rate(result_validation, obtener_resultados_reales(validation_set_original) )
            
    #-------------Prueba final con test set--------------------------------------------
    result_training, result_validation = obtener_resultados(k_fold_examples, test_set, vector_random, modelo)
    respuestas_obtenidas = result_validation + result_training
    #Si es una red (rn) entonces ya en result_training y result_validation tengo el error rate
    if modelo.rn:
        final_error_t = result_training
        final_error_v = result_validation
            
    #Si no, hay que calcularlo 
    else:
        final_error_t = (obtener_error_rate(result_training, obtener_resultados_reales(k_fold_examples_original) )/ len(k_fold_examples_original)) * 100
        final_error_v = (obtener_error_rate(result_validation, obtener_resultados_reales(test_set_original) ) / len(test_set_original)) * 100
    #-------------------------------------------------------------------------------

    return respuestas_obtenidas,(fold_error_t/len(k_fold_examples))*100, (fold_error_v/len(k_fold_examples))*100, final_error_t, final_error_v

