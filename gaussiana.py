#!/bin/python3
# Ian Mu;oz Nu;ez - Funcion 'Gaussiana'

import numpy as np
import matplotlib.pyplot as plt

v = np.linspace(-10, 10, 1000) # Variable independiente

A = 1 # Parametro que determina la magnitud de la funcion
B = 0.1 # Parametro que determina el ancho de la campana
y = A*np.exp(-B*v**2) # Funcion o variable dependiente
dy = -2*A*B*v*np.exp(-B*v**2) # Derivada de la funcion

fig, ax = plt.subplots(2,1, sharex=True)

ax[0].grid(True)
ax[0].plot(v, y, 'g-', linewidth=2)
ax[0].set_title("Funcion Gaussiana", fontsize=20)
ax[0].set_ylabel('$\\varphi(v)$', fontsize=15)

ax[1].grid(True)
ax[1].plot(v, dy, 'r-', linewidth=2)
ax[1].set_xlabel('v', fontsize=15)
ax[1].set_ylabel('$\\varphi\'(v)$', fontsize=15)
plt.show()

