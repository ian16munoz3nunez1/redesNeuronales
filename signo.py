#!/bin/python3
# Ian Mu;oz Nu;ez - Funcion 'Signo'

import numpy as np
import matplotlib.pyplot as plt

n = 1000
v = np.linspace(-10, 10, n) # Variable independiente

y = np.zeros(n)
for i in range(n):
    if v[i] > 0:
        y[i] = 1
    elif v[i] == 0:
        y[i] = 0
    else:
        y[i] = -1
dy = np.zeros(n) # Derivada de la funcion

fig, ax = plt.subplots(2,1, sharex=True)

ax[0].grid(True)
ax[0].plot(v, y, 'g-', linewidth=2)
ax[0].set_title("Funcion signo", fontsize=20)
ax[0].set_ylabel('$\\varphi(v)$', fontsize=15)

ax[1].grid(True)
ax[1].plot(v, dy, 'r-', linewidth=2)
ax[1].set_xlabel('v', fontsize=15)
ax[1].set_ylabel('$\\varphi\'(v)$', fontsize=15)

plt.show()

