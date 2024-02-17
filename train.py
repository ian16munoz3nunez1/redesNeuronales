#!/bin/python3

# Ian Mu;oz Nu;ez - MLP (Perceptron Multicapa)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mlp import MLP
pi = np.pi

n = 200
x = np.array([np.linspace(0, 2*pi, n)])
y = np.sin(x)

nn = MLP([(1,), (10, 'tanh'), (3, 'tanh'), (5, 'tanh'), (1,'linear')]) # Objeto de tipo Multi-Layer Perceptron
loss = nn.fit(x, y, 1e-2, 50000)
yp = nn.predict(x)

plt.figure(1)
plt.grid()

plt.plot(x.ravel(), y.ravel(), 'b-', linewidth=2)
plt.plot(x.ravel(), yp.ravel(), 'r--', linewidth=2)

plt.figure(2)
plt.grid()

plt.plot(loss, 'g-', linewidth=2)

plt.show()

