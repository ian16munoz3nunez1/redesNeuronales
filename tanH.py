#!/bin/python3
# Ian Mu;oz Nu;ez - Funcion 'Tangente Hiperbolica'

import numpy as np
import matplotlib.pyplot as plt

v = np.linspace(-10, 10, 1000) # Variable independiente

phi = np.tanh(v) # Funcion o variable dependiente

plt.figure(1)
plt.grid()

plt.plot(v, phi, 'g-', linewidth=2)

plt.title("Funcion tangente hiperbolica", fontsize=20)
plt.xlabel('v', fontsize=15)
plt.ylabel('$\phi(v)$', fontsize=15)

plt.show()

