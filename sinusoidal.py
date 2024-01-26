#!/bin/python3
# Ian Mu;oz Nu;ez - Funcion 'Sinusoidal'

import numpy as np
import matplotlib.pyplot as plt

v = np.linspace(-10, 10, 1000) # Variable independiente

A = 1 # Parametro que determina la magnitud de la funcion
B = 1 # Parametro que determina la frecuencia de la funcion
C = 0 # Parametro que determina la fase de la funcion
y = A*np.sin(B*v + C) # Funcion o variable dependiente
dy = A*np.cos(B*v + C) # Derivada de la funcion

fig, ax = plt.subplots(2,1, sharex=True)

ax[0].grid(True)
ax[0].plot(v, y, 'g-', linewidth=2)
ax[0].set_title("Funcion sinusoidal", fontsize=20)
ax[0].set_ylabel('$\\varphi(v)$', fontsize=15)

ax[1].grid(True)
ax[1].plot(v, dy, 'r-', linewidth=2)
ax[1].set_xlabel('v', fontsize=15)
ax[1].set_ylabel('$\\varphi\'(v)$', fontsize=15)

plt.show()

