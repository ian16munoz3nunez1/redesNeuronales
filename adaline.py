#!/bin/python3

# Ian Mu;oz Nu;ez - Adaline
# Desarrollar un código en el que un Adaline sea capaz de aproximar una
# función generada con datos aleatorios utilizando ahora el método
# mBGD (mini-Batch Gradient Descent), esta generaliza los metodos SGD
# y BGD, pues dependiendo el tamaño de los mini-lotes que se elige, la
# neurona se puede entrenar con un algoritmo parecido a un SGD o BGD
# (si se selecciona un tamaño de lote de 1, el algoritmo se comporta
# como SGD, si se elige un tamaño de lote del número de patrones de
# entrada, el algoritmo funcionara como un BGD.

import numpy as np

class Adaline:
    # Funcion para generar los lotes de los patrones de entrada
    def batcher(self, x, y, batch_size):
        p = x.shape[0]
        xl, xu = 0, batch_size # Indice inferior y superior para dividir por lotes

        while True:
            if xl < p:
                yield x[xl:xu], y[xl:xu] # Lotes de 'x' y 'y'
                xl, xu = xl+batch_size, xu+batch_size # Incremento de los indices para los lotes
            else:
                return None

    # Funcion para entrenar el Adaline
    def fit(self, x, y, eta, epocas, batch_size):
        self.w = np.random.rand() # Pesos sinapticos
        self.b = np.random.rand() # Bias

        p = x.shape[0] # Numero de patrones de entrada
        J = np.zeros(epocas) # Arreglo para almacenar el error del Adaline

        for epoca in range(epocas):
            minibatch = self.batcher(x, y, batch_size) # Division de la entrada por lotes
            for mx, my in minibatch:
                yp = np.dot(self.w, mx) + self.b # Interaccion de la entrada con los pesos y el bias

                e = my - yp # Error entre la salida deseada y la obtenida
                self.w += (eta/p) * np.dot(e, mx) # Ajuste de los pesos sinapticos
                self.b += (eta/p) * np.sum(e) # Ajuste del valor del bias

            J[epoca] = np.sum((y - (np.dot(self.w, x)+self.b))**2) # Error cuadratico medio

        return J

    # Funcin para generar una salida con un patron de entrada
    def predict(self, x):
        p = x.shape[0] # Numero de patrones de entrada
        yp = np.zeros(p) # Salida obtenida

        for i in range(p):
            yp[i] = np.dot(self.w, x[i]) + self.b # Interaccion de la entrada con los pesos y el bias

        return yp

