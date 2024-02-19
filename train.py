#!/bin/python3

# Ian Mu;oz Nu;ez - MLP (Perceptron Multicapa)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mlp import MLP
pi = np.pi

# print(list(colormaps)) # Funcion para listar los espacios de color de matplotlib

clases = 6 # Numero de clases deseadas
p = 40 # Numero de patrones de entrada
x = np.zeros((2, p*clases)) # Patrones de entrada
y = np.zeros((clases, p*clases)) # Salida deseada

# Llenado de los patrones de entrada y la salida deseada
xl, xu = -2, 2
for i in range(clases):
    seed = xl + (xu-xl)*np.random.rand(2,1)
    x[:, p*i:p*(i+1)] = seed + 0.2*np.random.rand(2,p)
    y[i, p*i:p*(i+1)] = np.ones((1, p))

nn = MLP([(2,), (10, 'tanh'), (20, 'tanh'), (6,'softmax')]) # Objeto de tipo Multi-Layer Perceptron
loss = nn.fit(x, y, 1e-2, 5000) # Entrenamiento de la red
yp = nn.predict(x) # Prediccion de la red

colors = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1]] # Colores para diferenciar los grupos

fig, ax = plt.subplots(1,3)

ax[0].grid()
yc = np.argmax(y, axis=0)
for i in range(x.shape[1]):
    ax[0].plot(x[0,i], x[1,i], 'o', c=colors[yc[i]], markersize=8)
ax[0].set_title("Problema original", fontsize=20)

ax[1].grid()
yc = np.argmax(yp, axis=0)
for i in range(x.shape[1]):
    ax[1].plot(x[0,i], x[1,i], 'o', c=colors[yc[i]], markersize=8)
ax[1].set_title("Prediccion de la red", fontsize=20)

n = 100
X, Y = np.meshgrid(np.linspace(xl, xu, n), np.linspace(xl, xu, n))
x = np.array([X.ravel(), Y.ravel()])
yp = nn.predict(x)
ax[2].grid()
yc = np.argmax(yp, axis=0)
for i in range(x.shape[1]):
    ax[2].plot(x[0,i], x[1,i], 'o', c=colors[yc[i]], markersize=4)
ax[2].set_title("Areas de clasificacion", fontsize=20)

plt.figure(2)
plt.grid()

plt.plot(loss, 'g-', linewidth=2)
plt.title("Grafica del error", fontsize=20)
plt.xlabel('Epocas', fontsize=15)
plt.ylabel('Error', fontsize=15)

plt.show()

