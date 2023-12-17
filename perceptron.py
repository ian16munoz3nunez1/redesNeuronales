#!/bin/python3

# Ian Mu;oz Nu;ez - Perceptron
# Ejemplo: Si se quiere entrenar un perceptron con
# una funcion AND de dos entradas, se tendria:
#       +-------------------+
#       | x_1 | x_2 | b | d |
#       +-----+-----+---+---+
#       |  0  |  0  | 1 | 0 | <-- Primer patron
#       +-----+-----+---+---+
#       |  0  |  1  | 1 | 0 | <-- Segundo patron
#       +-----+-----+---+---+
#       |  1  |  0  | 1 | 0 | <-- Tercer patron
#       +-----+-----+---+---+
#       |  1  |  1  | 1 | 1 | <-- Cuarto patron
#       +-----+-----+---+---+
# En donde 'x_1' es la entrada 1, 'x_2', la entrada 2, 'b'
# es la entrada del bias, que siempre está en 1 y 'd' son
# los valores deseados.

import numpy as np

class Perceptron:
    # Funcion de activacion tipo escalon
    def escalon(self, v):
        return 1 if v >= 0 else 0

    # Funcion para entrenar el perceptron
    def fit(self, x, y, epocas):
        self.w = np.random.rand(x.shape[0], 1) # Pesos sinapticos
        self.b = np.random.rand() # Bias

        p = x.shape[1] # Numero de patrones
        v = np.zeros(p)
        yp = np.zeros(p)
        e = np.zeros(p)

        for epoca in range(epocas):
            ep = 0 # Contador para los errores de la red
            for i in range(p):
                v[i] = np.dot(self.w.T, x[:,i]) + self.b # Interaccion de la entrada con los pesos y el umbral
                yp[i] = self.escalon(v[i]) # Salida obtenida con la funcion escalon

                e[i] = y[i] - yp[i] # Error obtenido
                if e[i] != 0:
                    self.w += e[i]*x[:,i].reshape(-1,1) # Ajuste de pesos sinapticos
                    self.b += e[i] # Ajuste del valor del bias
                    ep += 1 # Incremento del contador de errores

            # Si el perceptron no tuvo errores, se termina el entrenamiento
            if ep == 0:
                break

        return v, yp, e

    # Funcion para la generalizacion del perceptron
    def predict(self, x):
        p = x.shape[1] # Numero de patrones de entrada
        v = np.zeros(p)
        yp = np.zeros(p)

        for i in range(p):
            v[i] = np.dot(self.w.T, x[:,i]) + self.b # Interaccion de la entrada con los pesos y el umbral
            yp[i] = self.escalon(v[i]) # Salida obtenida con la funcion escalon

        return yp # Se regresa la salida del perceptron
