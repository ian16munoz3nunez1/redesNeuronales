#!/bin/python3

# Ian Mu;oz Nu;ez - Perceptron Unicapa
# Desarrollar un codigo en el que un perceptron unicapa sea
# capaz de clasificar multiples grupos utilizando la funcion
# softmax
#            z_j
#           e
# y_j = ----------
#         m
#        ---
#        \    z_i
#        /   e
#        ---
#        i=1

import numpy as np
import matplotlib.pyplot as plt
from olp import OLP

clases = 8 # Numero de clases deseadas
p = 80 # Numero de patrones por clase
x = np.zeros((2, p*clases)) # Patrones de entrada
y = np.zeros((clases, p*clases)) # Salida deseada

# Llenado de los patrones de entrada y la salida deseada
for i in range(clases):
    seed = np.random.rand(2,1)
    x[:, p*i:p*(i+1)] = seed + 0.15*np.random.rand(2,p)
    y[i, p*i:p*(i+1)] = np.ones((1,p))

a = 1 # Parametro para modificar el comportamiento de una funcion
eta = 1 # Factor de aprendizaje de la red
epocas = 1000 # Numero de iteraciones deseadas

nn = OLP(2, clases, 'softmax') # Objeto de tipo Perceptron Unicapa
nn.fit(x, y, eta, epocas) # Entrenamiento del Perceptron Unicapa
yp = nn.predict(x)[0] # Prediccion de los datos de entrada

colors = [[1,0,0], [0,1,0], [0,0,1], [0,1,1], [1,0,1], [1,1,0], [0,0,0], [0.5,0.5,0.5]] # Colores para diferenciar los grupos

fig, ax = plt.subplots(1,2)

ax[0].grid()
yc = np.argmax(y, axis=0)
for i in range(x.shape[1]):
    ax[0].plot(x[0,i], x[1,i], 'o', c=colors[yc[i]], markersize=8) # Grafica de los patrones de entrada
ax[0].set_title("Problema original", fontsize=20)

ax[1].grid()
yc = np.argmax(yp, axis=0)
for i in range(x.shape[1]):
    ax[1].plot(x[0,i], x[1,i], 'o', c=colors[yc[i]], markersize=8) # Grafica de los grupos clasificados
ax[1].set_title("OLP", fontsize=20)

plt.show()

