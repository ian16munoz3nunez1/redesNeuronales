#!/bin/python3

# Ian Mu;oz Nu;ez - MLP (Perceptron Multicapa)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mlp import MLP
pi = np.pi

df = pd.read_csv('regresionLineal1.csv')

x = np.array([df['x']])
y = np.array([df['y']])

nn = MLP([(1,), (1,'linear')])
print(f"Score: {nn.score(x, y)}")
loss = nn.fit(x, y, 1e-1, 200, batch_size=50)
print(f"Score: {nn.score(x, y)}")
yp = nn.predict(x)

plt.figure(1)
plt.grid()

plt.plot(x, y, 'bo', markersize=8)
plt.plot(x[0], yp[0], 'r-', linewidth=2)

plt.title("Regresion lineal", fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)

print(f"Loss: {loss[-1]}")

plt.figure(2)
plt.grid()

plt.plot(loss, 'g-', linewidth=2)

plt.title("Grafica del error", fontsize=20)
plt.xlabel('Epocas', fontsize=15)
plt.ylabel('Error', fontsize=15)

plt.show()

