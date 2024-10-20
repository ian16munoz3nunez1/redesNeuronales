#!python3

import numpy as np
import matplotlib.pyplot as plt
from rbf import RBF

xl = -5 # Limite inferior de la funcion
xu = 5 # Limite superior de la funcion
n = 10 # Numero de elementos de entrada

x = np.linspace(xl, xu, n).reshape(-1,1) # Datos de entrada
y = np.sin(x) # Salida deseada
plt.plot(x, y, 'ro')

k = 8 # Numero de nucleos
nn = RBF(k) # Modelo de la red neuronal

nn.fit(x,y,n) # Ajuste de los valores de la red

n = 200 # Numero de muestras nuevas
x = np.linspace(-5, 5, n).reshape(-1,1) # Datos de entrada
yp = nn.predict(x) # Prediccion de la red

plt.figure(1)
plt.grid()

plt.plot(x, yp, 'b-', linewidth=2)

plt.title("Funcion seno", fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)

plt.show()

