#!/bin/python3
# Ian Mu;oz Nu;ez - Funcion 'Escalon'

import numpy as np
import matplotlib.pyplot as plt

n = 1000
v = np.linspace(-10, 10, n) # Variable independiente

phi = np.zeros(n) # Funcion o variable dependiente
for i in range(n):
    if v[i] >= 0:
        phi[i] = 1
    else:
        phi[i] = 0

plt.figure(1)
plt.grid()

plt.plot(v, phi, 'g-', linewidth=2)

plt.title("Funcion escalon o umbral", fontsize=20)
plt.xlabel('v', fontsize=15)
plt.ylabel('$\\varphi(v)$', fontsize=15)

plt.show()

