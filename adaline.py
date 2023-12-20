#!/bin/python3

# Ian Mu;oz Nu;ez - Adaline
# Si se quiere entrenar un Adaline con una funcion AND de dos entradas, se tiene:
#       +--------------------+
#       | x_1 | x_2 | b |  d |
#       +-----+-----+---+----+
#       |  0  |  0  | 1 | -1 | <- Primer patron
#       +-----+-----+---+----+
#       |  0  |  1  | 1 | -1 | <- Segundo patron
#       +-----+-----+---+----+
#       |  1  |  0  | 1 | -1 | <- Tercer patron
#       +-----+-----+---+----+
#       |  1  |  1  | 1 |  1 | <- Cuarto patron
#       +--------------------+
# En donde 'x_1' es la entrada 1, 'x_2' la entrada 2, 'b' es la entrada del bias
# que siempre esta a 1 y 'd' los valores deseados

import numpy as np

class Adaline:
    # Funcion de activacion tipo signo
    def signo(self, v):
        return 1 if v >= 0 else -1

    # Funcion para entrenar el Adaline
    def fit(self, x, y, eta, epocas):
        self.w = np.random.rand(x.shape[0], 1) # Pesos sinapticos
        self.b = np.random.rand() # Bias

        p = x.shape[1] # Numero de patrones de entrada
        yp = np.zeros(p) # Salida obtenida

        for epoca in range(epocas):
            ep = 0 # Contador para los errores de la red
            for i in range(p):
                yp[i] = np.dot(self.w.T, x[:,i]) + self.b # Interaccion de la entrada con los pesos y el bias

                e = y[i] - yp[i] # Error entre la salida deseada y la obtenida
                if e != 0:
                    self.w += eta*e*x[:,i].reshape(-1,1) # Ajuste de los pesos sinapticos
                    self.b += eta*e # Ajuste del valor del bias
                    ep += 1 # Incremento del contador de errores

            # Si el Adaline no tuvo errores, se termina el entrenamiento
            if ep == 0:
                break

        return yp

    def predict(self, x):
        p = x.shape[1] # Numero de patrones de entrada
        yp = np.zeros(p) # Salida obtenida

        for i in range(p):
            yp[i] = np.dot(self.w.T, x[:,i]) + self.b # Interaccion de la entrada con los pesos y el bias

        return yp

