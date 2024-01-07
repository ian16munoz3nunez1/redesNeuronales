#!/bin/python3

# Ian Mu;oz Nu;ez - Adaline
# Desarrollar un código en el que una neurona sea capaz de
# aproximar patrones de entrada con la función logística:
#           1
# y = -------------
#     1 + exp(-a*v)

# Importacion de modulos
import numpy as np
import matplotlib.pyplot as plt
from adaline import Adaline

xl, xm, xu, n = 0, 10, 20, 40
x = np.array([np.hstack((xl+(xm-xl)*np.random.rand(n), xm+(xu-xm)*np.random.rand(n)))]) # Patrones de entrada
y = np.array([np.hstack((np.zeros(n), np.ones(n)))]) # Salida deseada
eta = 1e-1 # Factor de aprendizaje
epocas = 1000 # Numero de iteraciones deseadas

nn = Adaline() # Objeto de tipo Adaline
J = nn.fit(x, y, eta, epocas) # Entrenamiento del Adaline
yp = nn.classify(x) # Prediccion de los datos de entrada

v = np.array([np.arange(xl, xu, 0.001)]) # Arreglo de valores para el hiperplano separador
f = nn.predict(v) # Funcion logistica que define a los puntos

plt.figure(1)
plt.grid()

plt.plot(x[0], y[0], 'bo', markersize=4) # Grafica de los datos de entrada y salida deseada
plt.plot(v[0], f[0], 'r--', linewidth=2) # Grafica del hiperplano separador

# Informacion de la grafica
plt.title("El Adaline como aproximador", fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.legend(['Datos de entrada', 'Hiperplano'])

plt.figure(2)
plt.grid()

plt.plot(J, 'g-', linewidth=2) # Grafica del error

# Informacion de la grafica
plt.title("Grafica del error", fontsize=20)
plt.xlabel('Epocas', fontsize=15)
plt.ylabel('Error', fontsize=15)
plt.legend(['error'])

plt.show()

