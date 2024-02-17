#!/bin/python3

# Ian Mu;oz Nu;ez - MLP (Perceptron Multicapa)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mlp import MLP
pi = np.pi

# print(list(colormaps)) # Funcion para listar los espacios de color de matplotlib

n = 50
xl, xu = 0, 2*pi
X, Y = np.meshgrid(np.linspace(xl, xu, n), np.linspace(xl, xu, n))
Z = 2*np.cos(X) - np.sin(Y)
x = np.array([X.ravel(), Y.ravel()])
y = Z.ravel()

nn = MLP([(2,), (10, 'tanh'), (20, 'tanh'), (1,'linear')]) # Objeto de tipo Multi-Layer Perceptron
loss = nn.fit(x, y, 1e-2, 5000)
yp = nn.predict(x)

plt.figure(1)
ax = plt.axes(projection='3d')

ax.plot_surface(X, Y, Z, cmap=plt.cm.viridis)
ax.set_title("2cos(x) - sin(y)", fontsize=20)
ax.set_xlabel('x', fontsize=15)
ax.set_ylabel('y', fontsize=15)
ax.set_zlabel('z', fontsize=15)

plt.figure(2)
ax = plt.axes(projection='3d')

ax.plot_surface(X, Y, yp.reshape(X.shape), cmap=plt.cm.viridis)
ax.set_title("Prediccion del MLP", fontsize=20)
ax.set_xlabel('x', fontsize=15)
ax.set_ylabel('y', fontsize=15)
ax.set_zlabel('z', fontsize=15)

plt.figure(3)
plt.grid()

plt.plot(loss, 'g-', linewidth=2)
plt.title("Grafica del error", fontsize=20)
plt.xlabel('Epocas', fontsize=15)
plt.ylabel('Error', fontsize=15)

plt.show()

