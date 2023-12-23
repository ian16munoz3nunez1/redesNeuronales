#!/bin/python3

# Ian Mu;oz Nu;ez - Adaline
# Desarrollar un código en el que un Adaline sea capaz de aproximar la función
# descrita por los puntos
# - (1.0, 0.5)
# - (1.5, 1.1)
# - (3.0, 3.0)
# - (-1.2, -1.0)
# Los datos para el entrenamiento del Adaline resultan de la siguiente manera
#       +-----------------+
#       |   x  | b |   y  |
#       +------+---+------+
#       |  1.0 | 1 |  0.5 |
#       +------+---+------+
#       |  1.5 | 1 |  1.1 |
#       +------+---+------+
#       |  3.0 | 1 |  3.0 |
#       +------+---+------+
#       | -1.2 | 1 | -1.0 |
#       +-----------------+
# con las posiciones en 'x' como los datos de entrada y las posiciones en 'y'
# como la salida deseada, y 'b' como la entrada fija del bias.

# Importacion de modulos
import numpy as np
import matplotlib.pyplot as plt
import pickle
from adaline import Adaline

x = np.array([1.0, 1.5, 3.0, -1.2]) # Datos de entrada
y = np.array([0.5, 1.1, 3.0, -1.0]) # Salida deseada
epocas = 100 # Numero de iteraciones deseadas
eta = 0.05 # Factor de aprendizaje

nn = Adaline() # Objeto de tipo Adaline
J = nn.fit(x, y, eta, epocas) # Entrenamiento del Adaline
yp = nn.predict(x) # Prediccion de los datos de entrada

w, b = nn.w, nn.b # Pesos y bias obtenidos por la red
xl, xu = min(x.ravel())-0.1, max(x.ravel())+0.1 # Limites inferior y superior
t = np.arange(xl, xu, 0.1) # Arreglo de valores para el hiperplano separador

px = np.array([0, -b/w]) # Posiciones en 'x' para los puntos que definen al hiperplano
py = np.array([b, 0]) # Posiciones en 'y' para los puntos que definen al hiperplano
m = (py[1]-py[0])/(px[1]-px[0]) # Pendiente del hiperplano
f = m*(t-px[0]) + py[0] # Funcion del hiperplano

plt.figure(1)
plt.grid()
plt.axis('equal') # Muestra la escala de los ejes igual
plt.axis([xl, xu, xl, xu]) # Limite de los ejes 'x' y 'y'

# Grafica de los datos de entrada y su clasificacion
plt.plot(x, y, 'bo', markersize=4) # Datos de entrada y salida deseada
plt.plot(px, py, 'go', markersize=4) # Puntos del hiperplano
plt.plot(x, yp, 'yo', markersize=4) # Prediccion
plt.plot(t, f, 'r-', linewidth=2) # Hiperplano

# Informacion de la grafica
plt.title("El Adaline como aproximador", fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.legend(['Datos de entrada y salida', '$Puntos del hiperplano$', 'Prediccion', 'Hiperplano'])

plt.figure(2)
plt.grid()

plt.plot(J, 'g-', linewidth=2) # Grafica del error

# Informacion de la grafica
plt.title("Grafica del error", fontsize=20)
plt.xlabel('Epocas', fontsize=15)
plt.ylabel('Error', fontsize=15)
plt.legend(['error'])

plt.show()

