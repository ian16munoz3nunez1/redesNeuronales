#!/bin/python3

# Ian Mu;oz Nu;ez - MLP (Perceptron Multicapa)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mlp import MLP
pi = np.pi

df = pd.read_csv('regresionNoLinealTemp.csv')
x = np.array([df['time']])
y = np.array([df['temp']])

nn = MLP([(1,), (8,'tanh'), (8,'tanh'), (64,'tanh'), (64,'tanh'), (1,'linear')], scaler=True)
xTrain, xTest, yTrain, yTest = nn.train_test_split(x, y, train_size=0.7)

loss = nn.fit(xTrain, yTrain, 8e-2, 100, batch_size=100)
yp = nn.predict(x)

print(f"Train score: {nn.score(xTrain, yTrain)}")
print(f"Test score: {nn.score(xTest, yTest)}")
print(f"Loss: {loss[-1]}")

plt.figure(1)
plt.grid()

plt.plot(xTrain, yTrain, 'bo', markersize=4)
plt.plot(x[0], yp[0], 'r-', linewidth=2)

plt.title("Entrenamiento", fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)

plt.figure(2)
plt.grid()

plt.plot(xTest, yTest, 'bo', markersize=4)
plt.plot(x[0], yp[0], 'r-', linewidth=2)

plt.title("Generalizacion", fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)

plt.figure(3)
plt.grid()

plt.plot(loss, 'g-', linewidth=2)

plt.title("Grafica del error", fontsize=20)
plt.xlabel('Epocas', fontsize=15)
plt.ylabel('Error', fontsize=15)

plt.show()

