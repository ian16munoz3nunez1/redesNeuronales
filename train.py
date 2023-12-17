#!/bin/python3

# Ian Mu;oz Nu;ez - Perceptron
# Ejemplo: Si se quiere entrenar un perceptron con
# una funcion AND de dos entradas, se tendria:
#       +-------------------+
#       | x_1 | x_2 | b | d |
#       +-----+-----+---+---+
#       |  0  |  0  | 1 | 0 | <-- Primer patron
#       +-----+-----+---+---+
#       |  0  |  1  | 1 | 0 | <-- Segundo patron
#       +-----+-----+---+---+
#       |  1  |  0  | 1 | 0 | <-- Tercer patron
#       +-----+-----+---+---+
#       |  1  |  1  | 1 | 1 | <-- Cuarto patron
#       +-----+-----+---+---+
# En donde 'x_1' es la entrada 1, 'x_2', la entrada 2, 'b'
# es la entrada del bias, que siempre está en 1 y 'd' son
# los valores deseados.

# Importacion de modulos
import numpy as np
import matplotlib.pyplot as plt
import pickle
from perceptron import Perceptron

x = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]]) # Datos de entrenamiento
y = np.array([0, 0, 0, 1]) # Salida deseada
epocas = 100 # Numero de iteraciones deseadas

nn = Perceptron() # Objeto de tipo Perceptron
yp = nn.fit(x, y, epocas)[1] # Entrenamiento del perceptron

w1, w2, b = nn.w[0], nn.w[1], nn.b # Pesos y bias obtenidos por la red
xl, xu = -0.1, 1.1 # Limites inferior y superior para mostrar la grafica
t = np.arange(xl, xu, 0.1) # Arreglo de valores para el hiperplano separador

m = -w1/w2 # Pendiente del hiperplano separador
b = -b/w2 # Coeficiente del hiperplano separador
f = m*t + b # Funcion del hiperplano
# f = -(w1/w2)*t - (b/w2) # Funcion del hiperplano

plt.figure(1)
plt.grid()
plt.axis('equal') # Muestra la escala de los ejes igual
plt.axis([xl, xu, xl, xu]) # Limite de los ejes 'x' y 'y'

# Grafica de los datos de entrada y su clasificacion
plt.plot(x[0, yp==1], x[1, yp==1], 'yo', markersize=12)
plt.plot(x[0, yp==0], x[1, yp==0], 'go', markersize=12)
plt.plot(t, f, 'r-', linewidth=2) # Grafica del hiperplano

# Informacion de la grafica
plt.title("and", fontsize=20)
plt.xlabel('A', fontsize=15)
plt.ylabel('B', fontsize=15)
plt.legend(['$c_0$', '$c_1$', 'Hiperplano separador'])

plt.show()

pickle.dump(nn, open('perceptron.sav', 'wb')) # Se guarda la informacion del perceptron en un archivo sav

