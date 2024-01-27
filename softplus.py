#!/bin/python3
# Ian Mu;oz Nu;ez - Funcion 'Softplus'

import numpy as np
import matplotlib.pyplot as plt

v = np.linspace(-10, 10, 1000) # Variable independiente

a = 1 # Parametro que determina la pendiente de la funcion
y = np.log(1 + np.exp(a*v)) # Funcion o variable dependiente
dy = 1/(1 + np.exp(-a*v)) # Derivada de la funcion

fig, ax = plt.subplots(2,1, sharex=True)

ax[0].grid(True)
ax[0].plot(v, y, 'g-', linewidth=2)
ax[0].set_title("Funcion Softplus", fontsize=20)
ax[0].set_ylabel('$\\varphi(v)$', fontsize=15)

ax[1].grid(True)
ax[1].plot(v, dy, 'r-', linewidth=2)
ax[1].set_xlabel('v', fontsize=15)
ax[1].set_ylabel('$\\varphi\'(v)$', fontsize=15)

plt.show()

