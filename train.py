#!/bin/python3

# Ian Mu;oz Nu;ez - MLP (Perceptron Multicapa)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mlp import MLP

# print(list(colormaps)) # Funcion para listar los espacios de color de matplotlib

def plot(x, y, z, X, Y, Z, loss=None):
    fig = plt.figure(1)

    # ax1 = fig.add_subplot(131, projection='3d')
    ax1 = fig.add_subplot(131)
    ax1.grid(True)
    ax1.contourf(X, Y, Z, 50, cmap=plt.cm.viridis)
    ax1.plot(x[z<0.5], y[z<0.5], 'bo', markersize=8)
    ax1.plot(x[z>=0.5], y[z>=0.5], 'ro', markersize=8)
    ax1.set_xlabel('x', fontsize=15)
    ax1.set_ylabel('y', fontsize=15)

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.grid(True)
    ax2.plot_surface(X, Y, Z, cmap=plt.cm.viridis)
    ax2.plot(x, y, z, 'ro', markersize=8)
    ax2.set_xlabel('x', fontsize=15)
    ax2.set_ylabel('y', fontsize=15)
    ax2.set_zlabel('z', fontsize=15)

    # ax3 = fig.add_subplot(133, projection='3d')
    ax3 = fig.add_subplot(133)
    ax3.grid(True)
    ax3.contour(X, Y, Z, 50, cmap=plt.cm.viridis)
    ax3.plot(x[z<0.5], y[z<0.5], 'bo', markersize=8)
    ax3.plot(x[z>=0.5], y[z>=0.5], 'ro', markersize=8)
    ax3.set_xlabel('x', fontsize=15)
    ax3.set_ylabel('y', fontsize=15)

    if loss is not None:
        plt.figure(2)
        plt.grid()
        plt.plot(loss, 'g-', linewidth=2)
        plt.xlabel('x', fontsize=15)
        plt.ylabel('y', fontsize=15)

    plt.show()

x = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]]) # Patrones de entrada
y = np.array([[0, 1, 1, 0]]) # Salida deseada

nn = MLP([(2,), (3,'tanh'), (1,'logistic')]) # Objeto de tipo Multi-Layer Perceptron

xl, xu = np.min(x)-1, np.max(x)+1
X, Y = np.meshgrid(np.linspace(xl,xu,100), np.linspace(xl,xu,100))
data = np.array([X.ravel(), Y.ravel()])

Z = nn.predict(data)
Z = Z.reshape(X.shape)
yp = nn.predict(x)
yp = yp.ravel()
plot(x[0,:], x[1,:], yp, X, Y, Z)

loss = nn.fit(x, y, 0.1, 5000)
Z = nn.predict(data)
Z = Z.reshape(X.shape)
yp = nn.predict(x)
yp = yp.ravel()
plot(x[0,:], x[1,:], yp, X, Y, Z, loss)

