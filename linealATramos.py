#!/bin/python3
# Ian Mu;oz Nu;ez - Funcion 'Lineal a Tramos'

import numpy as np
import matplotlib.pyplot as plt

n = 1000
v = np.linspace(-10, 10, n) # Variable independiente

phi = np.zeros(n) # Funcion o variable dependiente
for i in range(n):
    if v[i] >= 1:
        phi[i] = 1
    elif v[i] > 0 and v[i] < 1:
        phi[i] = v[i]
    else:
        phi[i] = 0

plt.figure(1)
plt.grid()

plt.plot(v, phi, 'g-', linewidth=2)

plt.title("Funcion lineal a tramos", fontsize=20)
plt.xlabel('v', fontsize=15)
plt.ylabel('$\phi(v)$', fontsize=15)

plt.show()

