#!/bin/python3
# Ian Mu;oz Nu;ez - Funcion 'Tangente Hiperbolica'

import numpy as np
import matplotlib.pyplot as plt

v = np.linspace(-10, 10, 1000) # Variable independiente

a = 1 # Parametro que determina la magnitud de la funcion
b = 1 # Parametro que determina la frecuencia de la funcion
c = 0 # Parametro que determina la fase de la funcion
y = a*np.tanh(b*v + c) # Funcion o variable dependiente
dy = (b/a)*(1-y)*(1+y)

fig, ax = plt.subplots(2,1, sharex=True)

ax[0].grid(True)
ax[0].plot(v, y, 'g-', linewidth=2)
ax[0].set_title("Funcion tangente hiperbolica", fontsize=20)
ax[0].set_ylabel('$\\varphi(v)$', fontsize=15)

ax[1].grid(True)
ax[1].plot(v, dy, 'r-', linewidth=2)
ax[1].set_xlabel('v', fontsize=15)
ax[1].set_ylabel('$\\varphi\'(v)$', fontsize=15)

plt.show()

