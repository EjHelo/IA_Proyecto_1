# IA Proyecto 1

Tecnológico de Costa Rica

Ingeniería en Computación

Proyecto #1 - Inteligencia Artificial

# Manual de uso

## Instalación  de los requerimientos:
Este proyecto está desarrollado en Python versión 3.6, es importante mencionar que  NO funciona en versiones de Python inferior a 3.

Para la ejecución del sistema, es necesaria la herramienta "pip" que es un sistema de gestión de paquetes, utilizado para instalar bibliotecas para Python y modulos enviados. Una vez instalado pip se podrán instalar todas las bibliotecas necesarias para ejecutar el proyecto. Estas bibliotecas son:

* **NumPy:** Utilizado en arrays y compatibilidad con otras bibliotecas.
* **Scikit-learn:** Es útil con el manejo de redes neuronales.
* **keras:** Utilizado para la creación de redes neuronales.
* **tensorflow:** Utilizado en regresión lineal y en redes neuronales como backend.
* **Pandas:** Utilizado para la normalización estandar de los datos.
* **Pytest:** Utilizado para hacer las pruebas unitarias de los funciones.

## Comandos:

Para instalar las bibliotecas **(Numpy, Scikit-learn, Pandas, Pytest)** se llama a un mismo comando con la diferencia en el nombre de las bibliotecas

    pip install nombre_bibloteca

 Para instalar Keras
 
    pip install Keras

Para instalar Tensorflow

    pip install Tensorflow

## Uso del sistema
Después de haber instalado todas las bibliotecas necesarias para el proyecto, se debe clonar el repositorio en la computadora y abrir una termial o línea de comandos para encontrar la ubicación de los archivos y poder ejecutarlos. 

Para ejecutar los algoritmos de clasificación se debe ingresar en la línea de comandos **python main.py** seguida de diferentes banderas. 

Este programa recibe varias banderas, las cuales tienen un nombre, una descripción y un rango permitido para poder ejecutrase. Estas banderas son:

| Símbolo               	| Explicación                | Rango                                            |
|-------------------------	|--------------------------------------------------------------	|----------------------------------------------------	|
| --arbol               	| Activa el árbol de decisión                                                                                   	| True o False                                                                                        	|
| --prefijo             	| Nombre del archivo csv generado                                     	| Cadena de caracteres                                                                                          	|
| --kfold               	| Prueba el kfold crossvalidation                                                	| True o False                     	| Número entero positivo                                                                            |
| --cantidad-k            | Son los k grupos en los que se dividirá el set de entrenamiento                                                                                                           	|
| --porcentaje-prueba   	| Es el porcentaje de pruebas que se guardará para la prueba final                                              	| Números mayor a 0 y menor a 100                                                                     	|
| --red-neuronal        	| Activa la red neuronal                                                                                        	| True o False                                                                                        	|
| --numero-capas        	| Selecciona el número de capas en la red neuronal                                                        	| Numero entero positivo                                                                              	|
| --unidades-por-capa   	| Selecciona el número de unidad por capa en la red neuronal                                              	| Número entero positivo                                                                              	|
| --funcion-activacion  	| Selecciona la función de activación en la red neuronal                                                  	| softmax, softplus, relu, sigmoid 	|
| --umbral-poda         	| Selecciona el umbral con el que se podara el árbol .                                                    	| Número entre 0 y 1.                                                                                 	|



## Pruebas Unitarias

Para ejecutar las pruebas unitarias se debe ir a la línea de comandos o terminal y ejecutar el comando

    python -m pytest
    
Muestras si las pruebas son válidas o no
