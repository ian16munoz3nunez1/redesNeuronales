#!python3

# Ian Mu;oz Nu;ez - RBF (Redes Neuronales de Base Radial)

import numpy as np
import matplotlib.pyplot as plt
from rbf import RBF

xl = -5 # Limite inferior
xu = 5 # Limite superior
n = 20 # Numero de elementos

x = np.linspace(xl, xu, n).reshape(1,-1) # Patron de entrada
y = 2*np.cos(x) + np.sin(3*x) + 5 # Salida deseada

plt.plot(x, y, 'ro')

k = 16 # Numero de nucleos
nn = RBF(k) # Modelo de la red neuronal
nn.fit(x, y, n) # Entrenamiento de la red

n = 200 # Numero de elementos
x = np.linspace(xl, xu, n).reshape(1,-1) # Patron de entrada
yp = nn.predict(x).reshape(1,-1) # Prediccion de la red

plt.figure(1)
plt.grid()

plt.plot(x[0], yp[0], 'b-', linewidth=2)

plt.title("Funcion Seno", fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)

plt.show()

