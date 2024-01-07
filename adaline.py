#!/bin/python3

# Ian Mu;oz Nu;ez - Adaline
# Desarrollar un código en el que una neurona sea capaz de
# aproximar patrones de entrada con la función logística:
#           1
# y = -------------
#     1 + exp(-a*v)

import numpy as np

class Adaline:
    # Funcion para entrenar el Adaline
    def fit(self, x, y, eta, epocas):
        xl, xu = -5, 0 # Limites inferior y superior para los pesos
        self.w = xl+(xu-xl)*np.random.rand(x.shape[0], y.shape[0]) # Pesos sinapticos
        xl, xu = 0, 5 # Limites inferior y superior para el bias
        self.b = xl+(xu-xl)*np.random.rand(1, y.shape[0]) # Bias
        self.a = 1 # Valor para definir la pendiente de la funcion logistica

        p = x.shape[1] # Numero de patrones de entrada
        J = np.zeros(epocas) # Arreglo para almacenar el error del Adaline

        for epoca in range(epocas):
            yp = self.predict(x) # Interaccion de la entrada con los pesos y el bias

            e = y - yp # Error entre la salida deseada y la obtenida
            self.w += (eta/p) * np.dot(e, x.T).reshape(-1,1) # Ajuste de los pesos sinapticos
            self.b += (eta/p) * np.sum(e) # Ajuste del valor del bias

            J[epoca] = np.sum((y - self.predict(x))**2) # Error cuadratico medio

        return J

    # Funcion para generar la salida con la funcion logistica
    def predict(self, x):
        v = np.dot(self.w.T, x) + self.b # Interaccion de la entrada con los pesos y el bias
        yp = 1/(1 + np.exp(-self.a*v)) # Funcion logistica aplicada al valor 'v'

        return yp

    # Funcion para clasificar la salida de la neurona logistica
    def classify(self, x):
        v = np.dot(self.w.T, x) + self.b # Interaccion de la entrada con los pesos y el bias
        yp = 1/(1 + np.exp(-self.a*v)) # Funcion logistica aplicada al valor 'v'

        return 1 * (yp > 0.5)

