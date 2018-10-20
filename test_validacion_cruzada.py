from validacion_cruzada import *

def test_obtener_error_rate():
	# Datos de entrada
	dato1 = ['texture_mean', 'type', 'area_mean','-1 _ 1', 'Maligno']
	dato2 = ['texture', 'type', 'area_mean','-2 _ 1', 'Maligno']
    	# Ejecución
	resultado = obtener_error_rate(dato1, dato2)
	# Aserción
	assert type(resultado) is int
	assert resultado == 2

def test_obtener_resultados_reales():
    # Datos de entrada
    dato1 = [['142', '1', '0', '-1 _ 1'],['143', '1', '0', '-1 _ 1'],['144', '1', '0', '-1 _ 1'],['559', '1', '0', '-1 _ 1']]
    # Ejecución
    resultado = obtener_resultados_reales(dato1)
    # Aserción
    assert type(resultado) is list
    assert resultado == ['-1 _ 1', '-1 _ 1', '-1 _ 1', '-1 _ 1']


#Recibe un training set y un validation set. Busca el tipo de learner que se va a usar en options, asi como otros parametros necesarios
    
#Retorna el training y validation set para un kfold cv
def test_particion_k():
    dato1 = [['520', '1', '0', '2 _ 3', '2 _ 3', '-1 _ 1'],['521', '0', '1', '-1 _ 1', '-1 _ 1', '-1 _ 1'],
             ['522', '1', '0', '-1 _ 1', '-1 _ 1', '-1 _ 1'],['523', '1', '0', '-1 _ 1', '-1 _ 1', '-1 _ 1'],
             ['524', '1', '0', '2 _ 3', '2 _ 3', '-1 _ 1'],['525', '1', '0', '-1 _ 1', '-1 _ 1', '-1 _ 1'],
             ['526', '1', '0', '2 _ 3', '2 _ 3', '-1 _ 1'],['527', '0', '1', '-1 _ 1', '-1 _ 1', '-1 _ 1'],
             ['528', '1', '0', '2 _ 3', '2 _ 3', '-1 _ 1'],['529', '1', '0', '-1 _ 1', '-1 _ 1', '-1 _ 1']]
    dato2 = 2
    dato3 = 3
    # Ejecución
    resultado1, resultado2  = particion_k(dato1, dato2,dato3)
    # Aserción
    assert type(resultado1) is list
    assert resultado1 == [['520', '1', '0', '2 _ 3', '2 _ 3', '-1 _ 1'],
                          ['521', '0', '1', '-1 _ 1', '-1 _ 1', '-1 _ 1'],
                          ['522', '1', '0', '-1 _ 1', '-1 _ 1', '-1 _ 1'],
                          ['523', '1', '0', '-1 _ 1', '-1 _ 1', '-1 _ 1'],
                          ['524', '1', '0', '2 _ 3', '2 _ 3', '-1 _ 1'],
                          ['525', '1', '0', '-1 _ 1', '-1 _ 1', '-1 _ 1'],
                          ['529', '1', '0', '-1 _ 1', '-1 _ 1', '-1 _ 1']]
    assert type(resultado2) is list
    assert resultado2 == [['526', '1', '0', '2 _ 3', '2 _ 3', '-1 _ 1'],
                          ['527', '0', '1', '-1 _ 1', '-1 _ 1', '-1 _ 1'],
                          ['528', '1', '0', '2 _ 3', '2 _ 3', '-1 _ 1']]


def test_particion_h():
    # Datos de entrada
    dato1 = [['520', '1', '0', '2 _ 3', '2 _ 3', '-1 _ 1'],['521', '0', '1', '-1 _ 1', '-1 _ 1', '-1 _ 1'],
             ['522', '1', '0', '-1 _ 1', '-1 _ 1', '-1 _ 1'],['523', '1', '0', '-1 _ 1', '-1 _ 1', '-1 _ 1'],
             ['524', '1', '0', '2 _ 3', '2 _ 3', '-1 _ 1'],['525', '1', '0', '-1 _ 1', '-1 _ 1', '-1 _ 1'],
             ['526', '1', '0', '2 _ 3', '2 _ 3', '-1 _ 1'],['527', '0', '1', '-1 _ 1', '-1 _ 1', '-1 _ 1'],
             ['528', '1', '0', '2 _ 3', '2 _ 3', '-1 _ 1'],['529', '1', '0', '-1 _ 1', '-1 _ 1', '-1 _ 1']]
    dato2 = 30
    # Ejecución
    resultado1, resultado2  = particion_h(dato1, dato2)
    # Aserción
    assert type(resultado1) is list
    assert resultado1 == [['523', '1', '0', '-1 _ 1', '-1 _ 1', '-1 _ 1'],
                          ['524', '1', '0', '2 _ 3', '2 _ 3', '-1 _ 1'],
                          ['525', '1', '0', '-1 _ 1', '-1 _ 1', '-1 _ 1'],
                          ['526', '1', '0', '2 _ 3', '2 _ 3', '-1 _ 1'],
                          ['527', '0', '1', '-1 _ 1', '-1 _ 1', '-1 _ 1'],
                          ['528', '1', '0', '2 _ 3', '2 _ 3', '-1 _ 1'],
                          ['529', '1', '0', '-1 _ 1', '-1 _ 1', '-1 _ 1']]
    assert type(resultado2) is list
    assert resultado2 == [['520', '1', '0', '2 _ 3', '2 _ 3', '-1 _ 1'],
                          ['521', '0', '1', '-1 _ 1', '-1 _ 1', '-1 _ 1'],
                          ['522', '1', '0', '-1 _ 1', '-1 _ 1', '-1 _ 1']]
    


'''def test_k_fold_cross_validation():
    # Datos de entrada
    dato1 = ''
    dato2 = ''
    dato3 = ''
    dato4 = ''
    dato5 = ''
    # Ejecución
    res1,res2,res3,res4,res5= k_fold_cross_validation(dato1, dato2,dato3,dato4,dato5)
    # Aserción
    assert type(res1) is list
    assert type(res2) is list
    assert type(res3) is list
    assert type(res4) is list
    assert type(res5) is list
    assert res1 == []
    assert res2 == []
    assert res3 == []
    assert res4 == []
    assert res5 == []'''
    
