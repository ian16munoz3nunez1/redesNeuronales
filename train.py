#!/bin/python3

# Ian Mu;oz Nu;ez - Perceptron
# Diseñar una red neuronal unicapa, con función tipo escalón, con dos
# entradas y dos salidas que sea capaz de clasificiar los siguientes
# diez puntos en el plano, los cuales pertenecen a cuatro grupos:
# Grupo 1: (0.1, 1.2), (0.7, 1.8), (0.8, 1.6)
# Grupo 2: (0.8, 0.6), (1.0, 0.8)
# Grupo 3: (0.3, 0.5), (0.0, 0.2), (-0.3, 0.8)
# Grupo 4: (-0.5, -1.5), (-1.5, -1.3)

# Importacion de modulos
import numpy as np
import matplotlib.pyplot as plt
import pickle
from perceptron import Perceptron

x = np.array([[0.1, 0.7, 0.8, 0.8, 1.0, 0.3, 0.0, -0.3, -0.5, -1.5],
              [1.2, 1.8, 1.6, 0.6, 0.8, 0.5, 0.2, 0.8, -1.5, -1.3]]) # Datos de entrenamiento
y = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
              [0, 0, 0, 1, 1, 0, 0, 0, 1, 1]]) # Salida deseada
epocas = 100 # Numero de iteraciones deseadas

nn = Perceptron() # Objeto de tipo Perceptron
yp = nn.fit(x, y, epocas)[1] # Entrenamiento del perceptron

xl, xu = x.ravel().min()-0.1, x.ravel().max()+0.1 # Limites inferior y superior para mostrar la grafica
t = np.arange(xl, xu, 0.1) # Arreglo de valores para los hiperplanos

f1 = -(nn.w[0,0]/nn.w[1,0])*t - (nn.b[0]/nn.w[1,0]) # Funcion del hiperplano 1
f2 = -(nn.w[0,1]/nn.w[1,1])*t - (nn.b[1]/nn.w[1,1]) # Funcion del hiperplano 2

plt.figure(1)
plt.grid()
plt.axis('equal') # Muestra la escala de los ejes igual
plt.axis([xl, xu, xl, xu]) # Limite de los ejes 'x' y 'y'

# Grafica de los datos de entrenamiento y su clasificacion
plt.plot(x[0, (yp[0,:]==1)*(yp[1,:]==1)], x[1, (yp[0,:]==1)*(yp[1,:]==1)], 'ro', markersize=8)
plt.plot(x[0, (yp[0,:]==0)*(yp[1,:]==1)], x[1, (yp[0,:]==0)*(yp[1,:]==1)], 'bo', markersize=8)
plt.plot(x[0, (yp[0,:]==1)*(yp[1,:]==0)], x[1, (yp[0,:]==1)*(yp[1,:]==0)], 'go', markersize=8)
plt.plot(x[0, (yp[0,:]==0)*(yp[1,:]==0)], x[1, (yp[0,:]==0)*(yp[1,:]==0)], 'yo', markersize=8)
plt.plot(t, f1, 'm-', linewidth=2) # Grafica del hiperplano separador 1
plt.plot(t, f2, 'c-', linewidth=2) # Grafica del hiperplano separador 2

# Informacion de la grafica
plt.title("Clasificacion de 4 grupos", fontsize=20)
plt.xlabel('A', fontsize=15)
plt.ylabel('B', fontsize=15)
plt.legend(['$c_0$', '$c_1$', '$c_2$', '$c_3$', 'Hiperplano 1', 'Hiperplano 2'])

plt.show()

pickle.dump(nn, open('perceptron.sav', 'wb')) # Se guarda la informacion del perceptron en un archivo sav

