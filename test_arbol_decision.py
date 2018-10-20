from arbol_decision import *

def test_frecuencia():
	# Datos de entrada
	dato1 = [['-1 _ 1', 'Benigno', '2 _ 3'],['-1 _ 1', 'Maligno', '2 _ 3'],['<-2', 'Benigno', '2 _ 3']]
	dato2 = {}
	dato3 = 1
	# Ejecución
	resultado = frecuencia(dato1, dato2, dato3)
	# Aserción
	assert type(resultado) is dict
	assert resultado == {'Benigno': 2.0, 'Maligno': 1.0}

def test_valor_mayoria():
	# Datos de entrada
	dato1 = ['texture_mean', 'type', 'area_mean']
	dato2 = [['-1 _ 1', 'Benigno', '2 _ 3'],['-1 _ 1', 'Maligno', '2 _ 3'],['<-2', 'Benigno', '2 _ 3']]
	dato3 = 'type'
	# Ejecución
	resultado = valor_mayoria(dato1, dato2, dato3)
	# Aserción
	assert type(resultado) is str
	assert resultado == 'Benigno'

def test_calcular_entropia():
	# Datos de entrada
	dato1 = ['texture_mean', 'type', 'area_mean']
	dato2 = [['-1 _ 1', 'Benigno', '2 _ 3'],['-1 _ 1', 'Maligno', '2 _ 3'],['<-2', 'Benigno', '2 _ 3']]
	dato3 = 'type'
	# Ejecución
	resultado = calcular_entropia(dato1, dato2, dato3)
	# Aserción
	assert type(resultado) is float
	assert resultado == 0.9182958340544896

def test_ganancia_informacion():
	# Datos de entrada
	dato1 = ['texture_mean', 'type', 'area_mean']
	dato2 = [['-1 _ 1', 'Benigno', '2 _ 3'],['-1 _ 1', 'Maligno', '2 _ 3'],['<-2', 'Benigno', '2 _ 3']]
	dato3 = 'type'
	dato4 = 'texture_mean'
	# Ejecución
	resultado = ganancia_informacion(dato1, dato2, dato3, dato4)
	# Aserción
	assert type(resultado) is float
	assert resultado == 0.2516291673878229

def test_escoger_informacion():
	# Datos de entrada
	dato1 = [['-1 _ 1', 'Benigno', '2 _ 3'],['-1 _ 1', 'Maligno', '2 _ 3'],['<-2', 'Benigno', '2 _ 3']]
	dato2 = ['texture_mean', 'type', 'area_mean']
	dato3 = 'type'
	# Ejecución
	resultado = escoger_informacion(dato1, dato2, dato3)
	# Aserción
	assert type(resultado) is tuple
	assert len(resultado) == 2
	assert type(resultado[0]) is str
	assert type(resultado[1]) is float
	assert resultado == ('texture_mean', 0.2516291673878229)

def test_obtener_valores():
	# Datos de entrada
	dato1 = [['-1 _ 1', 'Benigno', '2 _ 3'],['-1 _ 1', 'Maligno', '2 _ 3'],['<-2', 'Benigno', '2 _ 3']]
	dato2 = ['texture_mean', 'type', 'area_mean']
	dato3 = 'type'
	# Ejecución
	resultado = obtener_valores(dato1, dato2, dato3)
	# Aserción
	assert type(resultado) is list
	assert len(resultado) == 2
	assert type(resultado[0]) is str
	assert type(resultado[1]) is str
	assert resultado == ['Benigno', 'Maligno']

def test_obtener_filas_validas():
	# Datos de entrada
	dato1 = [['-1 _ 1', 'Benigno', '2 _ 3'],['-1 _ 1', 'Maligno', '2 _ 3'],['<-2', 'Benigno', '2 _ 3']]
	dato2 = ['texture_mean', 'type', 'area_mean']
	dato3 = 'type'
	dato4 = 'Benigno'
	# Ejecución
	resultado = obtener_filas_validas(dato1, dato2, dato3, dato4)
	# Aserción
	assert type(resultado) is list
	assert len(resultado) == 2
	assert type(resultado[0]) is list
	assert type(resultado[1]) is list
	assert resultado == [['-1 _ 1', '2 _ 3'], ['<-2', '2 _ 3']]
      
def test_crear_arbol():
	# Datos de entrada        
	datos, atributos = read_csv("DataTree.csv")
	target = 'Dormir'
	# Ejecución
	resultado = crear_arbol(datos, atributos, target)
	# Aserción
	assert type(resultado) is dict
	assert resultado == {'Futbol': {'GanInfo': 0.38093714146565616, 'ValMay': 'Si', 'No': {'Clima': {'GanInfo': 0.6500224216483541, 'ValMay': 'Si', 'Soleado': 'Si', 'Lluvioso': 'Si', 'Ventoso': 'No'}}, 'Si': 'No', 'MM': {'Clima': {'GanInfo': 1.0, 'ValMay': 'No', 'Soleado': 'No', 'Ventoso': 'Si'}}}}
	#assert resultado == {'Futbol': {'GanInfo': 0.38093714146565616, 'ValMay': 'Si', 'No': {'Clima': {'GanInfo': 0.6500224216483541, 'ValMay': 'Si', 'Soleado': 'Si', 'Lluvioso': 'Si', 'Ventoso': 'No'}}, 'Si': 'No', 'MM': {'Clima': {'GanInfo': 1.0, 'ValMay': 'Si', 'Soleado': 'No', 'Ventoso': 'Si'}}}}

def test_es_hoja():
	# Datos de entrada
	arbol = {'Clima': {'GanInfo': 1.0, 'ValMay': 'Si', 'Soleado': 'No', 'Ventoso': 'Si'}}
	# Ejecución
	resultado = es_hoja(arbol)
	# Aserción
	assert type(resultado) is bool 
	assert resultado == True

def test_podar_arbol():
	# Datos de entrada
	datos, atributos = read_csv("DataTree.csv")
	target = 'Dormir'
	# Ejecución
	resultado = crear_arbol(datos, atributos, target)
	podar_arbol(resultado, 0.05)
	# Aserción
	assert type(resultado) is dict
	assert resultado == {'Futbol': {'GanInfo': 0.38093714146565616, 'ValMay': 'Si', 'No': {'Clima': {'GanInfo': 0.6500224216483541, 'ValMay': 'Si', 'Soleado': 'Si', 'Lluvioso': 'Si', 'Ventoso': 'No'}}, 'Si': 'No', 'MM': {'Clima': {'GanInfo': 1.0, 'ValMay': 'Si', 'Soleado': 'No', 'Ventoso': 'Si'}}}}

def test_prediccion_arbol_decision():
	# Datos de entrada
	arbol = {'Clima': {'GanInfo': 1.0, 'ValMay': 'Si', 'Soleado': 'No', 'Ventoso': 'Si'}}
	# Ejecución
	resultado = prediccion_arbol_decision(arbol, {'Clima':'Soleado'})
	# Aserción
	assert type(resultado) is str 
	assert resultado == 'No'
