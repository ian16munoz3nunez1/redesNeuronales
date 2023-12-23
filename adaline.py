#!/bin/python3

# Ian Mu;oz Nu;ez - Adaline
# Desarrollar un código en el que un Adaline sea capaz de aproximar la función
# descrita por los puntos
# - (1.0, 0.5)
# - (1.5, 1.1)
# - (3.0, 3.0)
# - (-1.2, -1.0)
# Los datos para el entrenamiento del Adaline resultan de la siguiente manera
#       +-----------------+
#       |   x  | b |   y  |
#       +------+---+------+
#       |  1.0 | 1 |  0.5 |
#       +------+---+------+
#       |  1.5 | 1 |  1.1 |
#       +------+---+------+
#       |  3.0 | 1 |  3.0 |
#       +------+---+------+
#       | -1.2 | 1 | -1.0 |
#       +-----------------+
# con las posiciones en 'x' como los datos de entrada y las posiciones en 'y'
# como la salida deseada, y 'b' como la entrada fija del bias.

import numpy as np

class Adaline:
    # Funcion de activacion tipo signo
    def signo(self, v):
        return 1 if v >= 0 else -1

    # Funcion para entrenar el Adaline
    def fit(self, x, y, eta, epocas):
        self.w = np.random.rand() # Pesos sinapticos
        self.b = np.random.rand() # Bias

        p = x.shape[0] # Numero de patrones de entrada
        yp = np.zeros(p) # Salida obtenida
        J = np.zeros(epocas) # Arreglo para almacenar el error del Adaline

        for epoca in range(epocas):
            permutation = np.random.permutation(p)
            x = x[permutation]
            y = y[permutation]

            for i in range(p):
                k = permutation[i]
                yp[k] = np.dot(self.w, x[k]) + self.b # Interaccion de la entrada con los pesos y el bias

                e = y[k] - yp[k] # Error entre la salida deseada y la obtenida
                self.w += eta*e*x[k] # Ajuste de los pesos sinapticos
                self.b += eta*e # Ajuste del valor del bias

            J[epoca] = np.sum((y - (np.dot(self.w,x) + self.b))**2) # Error minimo cuadrado

        return J

    def predict(self, x):
        p = x.shape[0] # Numero de patrones de entrada
        yp = np.zeros(p) # Salida obtenida

        for i in range(p):
            yp[i] = np.dot(self.w, x[i]) + self.b # Interaccion de la entrada con los pesos y el bias

        return yp

