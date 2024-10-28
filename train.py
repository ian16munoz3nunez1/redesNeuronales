#!python3

# Ian Mu;oz Nu;ez - RBF (Redes Neuronales de Base Radial)

import numpy as np
import matplotlib.pyplot as plt
from rbf import RBF

pi = np.pi
xl = 0 # Limite inferior de la funcion
xu = 1 # Limite superior de la funcion
n = 500 # Numero de elementos en el patron de entrada

x1 = xl + (xu-xl) * np.random.rand(1,n)
x2 = xl + (xu-xl) * np.random.rand(1,n)
x = np.vstack((x1, x2)) # Patron de entrada
y = (1.3356 * (1.5 * (1 - x1)) + (np.exp(2 * x1 - 1) * np.sin(3 * pi * (x1 - 0.6)**2)) + (np.exp(3 * (x2 - 0.5)) * np.sin(4 * pi * (x2 - 0.9)**2))) # Salida deseada

k = 16 # Numero de nucleos
nn = RBF(k) # Modelo de la red neuronal

nn.fit(x,y,n) # Ajuste de los valores de la red

yp = nn.predict(x) # Prediccion de la red

plt.figure(1)
plt.grid()

plt.plot(y[0], 'b-', linewidth=2)
plt.plot(yp.T[0], 'r-', linewidth=2)

plt.title("Random function", fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.legend(["function", "prediction"])

plt.show()

