#!/bin/python3

# Ian Mu;oz Nu;ez - Perceptron
# Diseñar una red neuronal unicapa, con función tipo escalón, con dos
# entradas y dos salidas que sea capaz de clasificiar los siguientes
# diez puntos en el plano, los cuales pertenecen a cuatro grupos:
# Grupo 1: (0.1, 1.2), (0.7, 1.8), (0.8, 1.6)
# Grupo 2: (0.8, 0.6), (1.0, 0.8)
# Grupo 3: (0.3, 0.5), (0.0, 0.2), (-0.3, 0.8)
# Grupo 4: (-0.5, -1.5), (-1.5, -1.3)

import numpy as np

class Perceptron:
    # Funcion de activacion tipo escalon
    def escalon(self, v):
        return 1 if v >= 0 else 0

    # Funcion para entrenar el perceptron
    def fit(self, x, y, epocas):
        self.w = np.random.rand(x.shape[0], y.shape[0]) # Pesos sinapticos
        self.b = np.random.rand(y.shape[0]) # Bias

        p = x.shape[1] # Numero de patrones
        v = np.zeros((y.shape[0], p))
        yp = np.zeros((y.shape[0], p))
        e = np.zeros((y.shape[0], p))

        for epoca in range(epocas):
            ep = 0 # Contador para los errores de la red
            for i in range(p):
                for j in range(y.shape[0]):
                    v[j,i] = np.dot(self.w[:,j].T, x[:,i]) + self.b[j] # Interaccion de la entrada con los pesos y el umbral
                    yp[j,i] = self.escalon(v[j,i]) # Salida obtenida con la funcion escalon

                    e[j,i] = y[j,i] - yp[j,i] # Error obtenido
                    if e[j,i] != 0:
                        self.w[:,j] += e[j,i]*x[:,i] # Ajuste de pesos sinapticos
                        self.b[j] += e[j,i] # Ajuste del valor del bias
                        ep += 1 # Incremento del contador de errores

            # Si el perceptron no tuvo errores, se termina el entrenamiento
            if ep == 0:
                break

        return v, yp, e

    # Funcion para la generalizacion del perceptron
    def predict(self, x):
        p = x.shape[1] # Numero de patrones de entrada
        v = np.zeros((self.w.shape[1], p))
        yp = np.zeros((self.w.shape[1], p))

        for i in range(p):
            for j in range(self.w.shape[1]):
                v[j,i] = np.dot(self.w[:,j].T, x[:,i]) + self.b[j] # Interaccion de la entrada con los pesos y el umbral
                yp[j,i] = self.escalon(v[j,i]) # Salida obtenida con la funcion escalon

        return yp # Se regresa la salida del perceptron

