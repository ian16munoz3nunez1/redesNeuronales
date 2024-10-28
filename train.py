#!python3

# Ian Mu;oz Nu;ez - RBF (Redes Neuronales de Base Radial)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from rbf import RBF

# print(list(colormaps)) # Muestra los espacios de color para la malla de matplotlib

pi = np.pi
xl = 0 # Limite inferior
xu = 2*pi # Limite superior
n = 20 # Numero de elementos

# Patron de entrada
x = np.linspace(xl, xu, n).reshape(1,-1)
y = np.linspace(xl, xu, n).reshape(1,-1)
xy = np.vstack((x,y))
X, Y = np.meshgrid(x, y)

# Salida deseada
Z = np.cos(X) - 3*np.sin(Y)

fig = plt.figure(1)
ax0 = fig.add_subplot(121, projection='3d')
ax0.plot_surface(X, Y, Z, cmap='viridis')
ax0.set_title("Funcion", fontsize=20)
ax0.set_xlabel('x', fontsize=15)
ax0.set_ylabel('y', fontsize=15)
ax0.set_zlabel('z', fontsize=15)
ax0.view_init(elev=20, azim=-135)

k = 12 # Numero de nucleos
nn = RBF(k) # Modelo de la red neuronal
nn.fit(xy, Z, n) # Entrenamiento de la red

zp = nn.predict(xy) # Prediccion de la red

ax1 = fig.add_subplot(122, projection='3d')
ax1.plot_surface(X, Y, zp.T, cmap='viridis')
ax1.set_title("Prediccion", fontsize=20)
ax1.set_xlabel('x', fontsize=15)
ax1.set_ylabel('y', fontsize=15)
ax1.set_zlabel('z', fontsize=15)
ax1.view_init(elev=20, azim=-135)

plt.show()

