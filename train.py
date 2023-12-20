#!/bin/python3

# Ian Mu;oz Nu;ez - Adaline
# Si se quiere entrenar un Adaline con una funcion AND de dos entradas, se tiene:
#       +--------------------+
#       | x_1 | x_2 | b |  d |
#       +-----+-----+---+----+
#       |  0  |  0  | 1 | -1 | <- Primer patron
#       +-----+-----+---+----+
#       |  0  |  1  | 1 | -1 | <- Segundo patron
#       +-----+-----+---+----+
#       |  1  |  0  | 1 | -1 | <- Tercer patron
#       +-----+-----+---+----+
#       |  1  |  1  | 1 |  1 | <- Cuarto patron
#       +--------------------+
# En donde 'x_1' es la entrada 1, 'x_2' la entrada 2, 'b' es la entrada del bias
# que siempre esta a 1 y 'd' los valores deseados

# Importacion de modulos
import numpy as np
import matplotlib.pyplot as plt
import pickle
from adaline import Adaline

x = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]]) # Datos de entrenamiento
y = np.array([-1, -1, -1, 1]) # Salida deseada
epocas = 100 # Numero de iteraciones deseadas
eta = 0.2 # Factor de aprendizaje

nn = Adaline() # Objeto de tipo Adaline
yp = nn.fit(x, y, eta, epocas) # Entrenamiento del Adaline
yp = np.array([nn.signo(v) for v in yp]) # Funcion signo aplicada a la salida del Adaline

w1, w2, b = nn.w[0], nn.w[1], nn.b # Pesos y bias obtenidos por la red
xl, xu = min(x.ravel())-0.1, max(x.ravel())+0.1 # Limites inferior y superior
t = np.arange(xl, xu, 0.1) # Arreglo de valores para el hiperplano separador

m = -w1/w2 # Pendiente del hiperplano
b = -b/w2 # Coeficiente del hiperplano
f = m*t + b # Funcion del hiperplano separador

plt.figure(1)
plt.grid()
plt.axis('equal') # Muestra la escala de los ejes igual
plt.axis([xl, xu, xl, xu]) # Limite de los ejes 'x' y 'y'

# Grafica de los datos de entrada y su clasificacion
plt.plot(x[0, yp==1], x[1, yp==1], 'yo', markersize=12)
plt.plot(x[0, yp==-1], x[1, yp==-1], 'go', markersize=12)
plt.plot(t, f, 'r-', linewidth=2) # Grafica del hiperplano

# Informacion de la grafica
plt.title("El Adaline como clasificador", fontsize=20)
plt.xlabel('A', fontsize=15)
plt.ylabel('B', fontsize=15)
plt.legend(['$c_0$', '$c_1$', 'Hiperplano'])

plt.show()

pickle.dump(nn, open('adaline.sav', 'wb')) # Se guarda la informacion del Adaline en un archivo sav

