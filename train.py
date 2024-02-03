#!/bin/python3

# Ian Mu;oz Nu;ez - Perceptron Unicapa
# Desarrollar un codigo en el que un perceptron unicapa sea
# capaz de clasificar 4 grupos utilizando la funcion
# logistica y el esquema One-vs-All
#           1
# y = -------------
#     1 + exp(-a*v)

import numpy as np
import matplotlib.pyplot as plt
from olp import OLP

seed = np.array([[0, 0, 1, 1], [0, 1, 0, 1]]) # Puntos de referencia para los grupos
clases = 4 # Numero de clases deseadas
p = 20 # Numero de patrones por clase
x = np.zeros((2, p*clases)) # Patrones de entrada
y = np.zeros((clases, p*clases)) # Salida deseada

# Llenado de los patrones de entrada y la salida deseada
for i in range(clases):
    x[:, p*i:p*(i+1)] = seed[:,i].reshape(2,1) + 0.15*np.random.rand(2,p)
    y[i, p*i:p*(i+1)] = np.ones((1, p))

a = 1 # Parametro para modificar el comportamiento de una funcion
eta = 1 # Factor de aprendizaje de la red
epocas = 1000 # Numero de iteraciones deseadas
nn = OLP(2, clases, 'logistic') # Objeto de tipo Perceptron Unicapa
nn.fit(x, y, eta, epocas) # Entrenamiento del Perceptron Unicapa
yp = nn.predict(x)[0] # Prediccion de los datos de entrada

xl, xu = x.ravel().min()-0.1, x.ravel().max()+0.1 # Limites inferior y superior para los hiperplanos
t = np.arange(xl, xu, 0.1) # Arreglo de valores para los hiperplanos
colors = [[1,0,0], [0,1,0], [0,0,1], [1,1,0]] # Colores para diferenciar los grupos

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

for i in range(clases):
    w1, w2, b = nn.w[0,i], nn.w[1,i], nn.b[i,0] # Pesos y bias de la red
    m = -w1/w2 # Pendiente del hiperplano
    b = -b/w2 # Coeficiente del hiperplano
    f = m*t + b # Funcion del hiperplano separador
    ax[1].plot(t, f, '-', c=colors[i]) # Grafica de los hiperplanos
ax[1].set_title("OLP", fontsize=20)

plt.show()

