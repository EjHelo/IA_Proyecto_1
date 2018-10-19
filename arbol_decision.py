import math
import random

'''Función que retorna el diccionario con la frecuencia de los datos'''
def frecuencia(datos, frecuencias, index):
    #recorre los datos para verificar que el indice de la fila esté en las frecuencias
    for fila in datos:
        if (fila[index] in frecuencias):
            frecuencias[fila[index]] += 1.0 #si está le suma uno
        else:
            frecuencias[fila[index]] = 1.0  #sino, le asigna uno
    return frecuencias

'''Funcion que busca los atributos mas comunes'''
def valor_mayoria(atributos, datos, target):
    valor_frecuencia = {}
    index = atributos.index(target) #Busca el target
    valor_frecuencia = frecuencia(datos, valor_frecuencia, index) #Calcula frecuencia de datos
    mayoria_A = True #el default de la variable es true
    maximo = 0.0 #asigna el mínimo a la variable
    mayor = ""
    #hace un ciclo que recorre las llaves de los valores de frecuencia 
    for llave in valor_frecuencia.keys():
        if valor_frecuencia[llave] > maximo:
            maximo = valor_frecuencia[llave]
            mayor = llave
            mayoria_A = True
        elif valor_frecuencia[llave] == maximo:
            mayoria_A = False

    #Si no existe mayoria se retorna un random
    if not mayoria_A:
        valores = list(valor_frecuencia.keys())
        rand = random.choice(valores)
        return rand

    return mayor

'''Funcion que calcula la entropia de los datos
   Retorna el valor de la entropia'''
def calcular_entropia(atributos, datos, Attr):
    entropia = 0.0 #le asigna el valor mínimo a la entropía
    valor_frecuencia = {}
    indice = 0 #selecciona index inicial
    
    for valor in atributos:
        if (Attr == valor):
            break
        indice += 1

    #calcula la frecuencia de los datos con la función frecuencia
    valor_frecuencia = frecuencia(datos, valor_frecuencia, indice)

    # Calcula la entropia recorriendo los valores de las frecuencias
    for freq in valor_frecuencia.values():
        #la fórmula de la entropía se calcula con la sumatoria de la
        #la frecuencia entre el total de los datos por el log2 de lo mismo 
        entropia += (-freq/len(datos)) * math.log(freq/len(datos), 2)


    return entropia

''' Retorna la ganancia de informacion del atributo
    Se calcula esa ganancia para determinar si hay que dividir con este atributo'''
def ganancia_informacion(atributos, datos, attr, targetAttr):
    valor_frecuencia = {}
    subset_entropia = 0.0
    
    indice = atributos.index(attr) #index del atributo

    # Se calcula las veces que aparece cada valor del atributo al que se le está calculando ganancia 
    valor_frecuencia = frecuencia(datos, valor_frecuencia, indice)

    # Se calcula la entropia para cada uno de los subgrupos 
    for valor in valor_frecuencia.keys():
        valor_probabilidad = valor_frecuencia[valor] / sum(valor_frecuencia.values())
        dataSubset =  []

        for fila in datos:
            if fila[indice] == valor:
                dataSubset.append(fila)
        #calculamos la entropía llamando a la función
        valor_entropia = calcular_entropia(atributos, dataSubset, targetAttr)
        subset_entropia += valor_probabilidad * valor_entropia #Resto

    entropia_total = calcular_entropia(atributos, datos, targetAttr)
    ganancia = entropia_total - subset_entropia #hace la resta de las entropías para la ganancia

    return ganancia

'''Función que escoge el mejor atributo
    Retorna ese mejor atributo con su ganancia'''
def escoger_informacion(datos, atributos, target):
    mejor = atributos[0] #asigna el primer atributo a la variable mejor 
    maxima_ganancia = 0
    for attr in atributos:
        if attr != target: 
            #si son diferentes calcula una nueva ganancia
            nueva_ganancia = ganancia_informacion(atributos, datos, attr, target) 
            if nueva_ganancia > maxima_ganancia:
                maxima_ganancia = nueva_ganancia
                mejor = attr
    if(maxima_ganancia > 1): maxima_ganancia = 1.0
    return mejor, maxima_ganancia

'''Retorna una lista con todos los posibles valores de un atributo'''
def obtener_valores(datos, atributos, attr):
    indice = atributos.index(attr)
    valores = []
    #ciclo que recorre las filas de los datos para ver si el indice está o no
    for fila in datos:
        if fila[indice] not in valores: 
            valores.append(fila[indice])#lo agrego a los valores si no se encontró
    
    return valores

'''Retorna las filas que a este nivel del arbol siguen siendo validas'''
def obtener_filas_validas(datos, atributos, mejor, val):
    filas_validas = [[]]
    indice = atributos.index(mejor)
    for valor in datos:
        if (valor[indice] == val):
            nueva_entrada = []

            #agrega el valor si no es el mejor
            for i in range(0,len(valor)):
                if(i != indice):
                    nueva_entrada.append(valor[i])
            filas_validas.append(nueva_entrada)
    filas_validas.remove([])
    return filas_validas

def crear_arbol(datos, atributos, target):
    
    datos = datos[:] #Copia del parametro data, no una referencia
    valores = [] 
    
    #Una lista con todos los valores actuales del target
    for fila in datos:
        valores.append(fila[atributos.index(target)])
    #Retorna cual atributo tiene más presencia, ese atributo será el default
    default = valor_mayoria(atributos, datos, target) 
    
    # Si el dataset esta limpio retorna el default value. 
    # Se verifica si aun quedan atributos, se resta - 1 para no tomar en cuenta el atributo target

    #Si los datos están vacíos
    if not datos or (len(atributos) - 1) <= 0:
        return default
    # Si todos los ejemplos tienen la misma clasificación entonces retorne esa clasificación
    if valores.count(valores[0]) == len(valores):
        return valores[0]
    else:
        # Elija el siguiente mejor atributo
        mejor, maxima_ganancia = escoger_informacion(datos, atributos, target)

        # Creamos un nuevo nodo con el mejor atributo
        arbol = {mejor:{'GanInfo':maxima_ganancia, 'ValMay':default}}

        # Verificamos si hay nodos que falten de agregar
        valores_actuales = obtener_valores(datos, atributos, mejor)

        # Creamos un nuevo nodo por cada posible valor del mejor atributo
        for valor in valores_actuales:
            filas_validas = obtener_filas_validas(datos, atributos, mejor, valor) #Retorna todos los ejemplos para un valor del mejor atributo
            nuevo_atributo = atributos[:] #Hace una copia de los atributos para no modificar los originales
            nuevo_atributo.remove(mejor) #Quita el mejor atributo
            sub_arbol = crear_arbol(filas_validas, nuevo_atributo, target) #Crea el nuevo nodo
            # Asigna el nuevo nodo
            arbol[mejor][valor] = sub_arbol
        
    return arbol

def prediccion_aux_arbol(arbol, input):
    llave_atributo = list(arbol.keys())[0]
    sub_arbol = arbol[llave_atributo]
    
    valor_entrada = input.get(llave_atributo,None)
    
    if(valor_entrada not in sub_arbol):
        # Es un nuevo caso no visto del branch
        return sub_arbol['ValMay']
    rama = sub_arbol[valor_entrada]

    if(type(rama) is dict):
        return prediccion_aux_arbol(rama, input)
    else:
        return rama

def prediccion_arbol_decision(arbol, input):
    arbol_temporal = arbol.copy()
    return prediccion_aux_arbol(arbol_temporal, input)


def es_hoja(arbol): 
    atributo_llave = list(arbol.keys())[0]
    sub_arbol = arbol[atributo_llave]
    for llave in sub_arbol:
        if(type(sub_arbol[llave]) is dict):
            return False
    return True

def podar_arbol(arbol, threshold):
    atributo_llave = list(arbol.keys())[0]
    sub_arbol = arbol[atributo_llave]
    if(es_hoja(arbol) is True):
        if(sub_arbol['GanInfo'] <= threshold):
            # debe ser podado
            return sub_arbol['ValMay']
        else:
            return None
    else:
        n_sub_arboles = 0
        n_sub_arboles_podados = 0
        for llave in sub_arbol:
            if(type(sub_arbol[llave]) is dict):
                n_sub_arboles += 1
                poda = podar_arbol(sub_arbol[llave], threshold)
                if(poda != None):
                    sub_arbol[llave] = poda
                    n_sub_arboles_podados +=1
        if(n_sub_arboles == n_sub_arboles_podados and n_sub_arboles != 0):
            if(sub_arbol['GanInfo'] <= threshold):
                # debe ser podado
                return sub_arbol['ValMay']
            else:
                return None

def construir_arbol_decision(nombre_archivo, target):
    archivo = open(nombre_archivo)
    datos = [[]]

    for linea in archivo:
        linea = linea.strip("\r\n")
        datos.append(linea.split(','))
    datos.remove([])
    atributos = datos[0]
    datos.remove(atributos)
    #Recibe una lista de listas, los atributos, y el atributo que vamos a querer predecir  
    return crear_arbol(datos, atributos, target)

'''
B_arboles = 4

if B_arboles > 1 and B_arboles <= 4:
    cantidad_atributos = 15
    
if B_arboles > 4 and B_arboles <= 8:
    cantidad_atributos = 10
    
if B_arboles > 8:
    cantidad_atributos = 5
    #my_randoms = [random.randrange(1,31,1) for _ in range(cantidad_atributos)]
    
####

random_general=[]
for _ in range (B_arboles):
    my_randoms = [random.randint(1,31) for _ in range(cantidad_atributos)]
    arbol_set= dataframe.iloc[:,my_randoms]
    arbol_set.to_csv('arbolSet.csv', sep=',')
    #llamar a construir arbol de decision 
    #target = ''
    #arbol = construir_arbol_decision('arbolSet.csv',target)
    random_general += [my_randoms]
'''
    
    
#print(random_general)
#sub_set= dataframe.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]
#set+=dataframe[ dataframe.columns[1]]
#sub_set.to_csv('subSet.csv', sep=',')


#print("Versión del Árbol en llaves\n\n", arbol)

#result = prediccion_arbol_decision(arbol, {'Dormir':'Si'})

#print("Prediccion del resultado es: ",result)
