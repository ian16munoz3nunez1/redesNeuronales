#!/bin/python3
# Ian Mu;oz Nu;ez - Funcion 'ReLU'

import numpy as np
import matplotlib.pyplot as plt

n = 1000
v = np.linspace(-10, 10, n) # Variable independiente

y = np.maximum(np.zeros(n), v) # Funcion o variable dependiente
dy = np.zeros(n) # Derivada de la funcion
for i in range(n):
    if v[i] >= 0:
        dy[i] = 1
    else:
        dy[i] = 0

fig, ax = plt.subplots(2,1, sharex=True)

ax[0].grid(True)
ax[0].plot(v, y, 'g-', linewidth=2)
ax[0].set_title("Funcion ReLU", fontsize=20)
ax[0].set_ylabel('$\\varphi(v)$', fontsize=15)

ax[1].grid(True)
ax[1].plot(v, dy, 'r-', linewidth=2)
ax[1].set_xlabel('v', fontsize=15)
ax[1].set_ylabel('$\\varphi\'(v)$', fontsize=15)

plt.show()

