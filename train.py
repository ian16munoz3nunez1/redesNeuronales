#!/bin/python3

# Ian Mu;oz Nu;ez - Adaline
# Desarrollar un código en el que una neurona sea capaz de
# clasificar patrones de entrada con una función AND
# utilizando la función logística:
#           1
# y = -------------
#     1 + exp(-a*v)

# Importacion de modulos
import numpy as np
import matplotlib.pyplot as plt
from adaline import Adaline

x = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]]) # Datos de entrenamiento
y = np.array([[0, 0, 0, 1]]) # Salida deseada
eta = 1e-1 # Factor de aprendizaje
epocas = 1000 # Numero de iteraciones deseadas

nn = Adaline() # Objeto de tipo Adaline
J = nn.fit(x, y, eta, epocas) # Entrenamiento del Adaline
yp = nn.classify(x) # Prediccion de los datos de entrada

w1, w2, b = nn.w[0,0], nn.w[1,0], nn.b # Pesos y bias obtenidos por la red
xl, xu = min(x.ravel())-0.1, max(x.ravel())+0.1 # Limites inferior y superior
v = np.array([np.arange(xl, xu, 0.1)]) # Arreglo de valores para el hiperplano separador

m = -w1/w2 # Pendiente del hiperplano
b = -b/w2 # Coeficiente del hiperplano
f = m*v + b # Funcion del hiperplano separador

plt.figure(1)
plt.grid()
plt.axis('equal') # Muestra la escala de los ejes igual
plt.axis([xl, xu, xl, xu]) # Limite de los ejes 'x' y 'y'

# Grafica de los datos de entrada y su clasificacion
plt.plot(x[0,yp[0]==1], x[1,yp[0]==1], 'yo', markersize=8)
plt.plot(x[0,yp[0]==0], x[1,yp[0]==0], 'go', markersize=8)
plt.plot(v[0], f[0], 'r-', linewidth=2) # Grafica del hiperplano separador

# Informacion de la grafica
plt.title("El Adaline como aproximador", fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.legend(['$c_0$', '$c_1$', 'Hiperplano'])

plt.figure(2)
plt.grid()

plt.plot(J, 'g-', linewidth=2) # Grafica del error

# Informacion de la grafica
plt.title("Grafica del error", fontsize=20)
plt.xlabel('Epocas', fontsize=15)
plt.ylabel('Error', fontsize=15)
plt.legend(['error'])

plt.show()

