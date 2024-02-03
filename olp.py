#!/bin/python3

# Ian Mu;oz Nu;ez - Perceptron Unicapa
# Desarrollar un codigo en el que un perceptron unicapa sea
# capaz de clasificar 4 grupos utilizando la funcion
# logistica y el esquema One-vs-All
#           1
# y = -------------
#     1 + exp(-a*v)

import numpy as np
from colorama import init
from colorama.ansi import Fore

init(autoreset=True)

class OLP:
    def __init__(self, n_inputs, n_outputs, function):
        functions = ['step', 'linear', 'logistic', 'softmax'] # Funciones para el la red
        # Si se elige una funcion no disponible se muestra un error
        if not function.lower() in functions:
            print(Fore.RED + f"Function \'{function}\' not in available functions")
            for i, f in enumerate(functions):
                print(Fore.YELLOW + f"{i}. {f}")
            exit(1)

        xl, xu = -2, 2 # Limites inferior y superior para los pesos y bias
        self.w = xl + (xu-xl)*np.random.rand(n_inputs,n_outputs) # Pesos sinapticos
        self.b = xl + (xu-xl)*np.random.rand(n_outputs,1) # Bias
        self.f = functions.index(function.lower()) # Indice de la funcion a usar

    def fit(self, x, d, eta, epocas, a=1):
        p = x.shape[1] # Numero de patrones de entrada

        for epoca in range(epocas):
            v = np.dot(self.w.T, x) + self.b # Interaccion de la entrada con los pesos y el bias
            yp, dyp = self.function(v, a) # Salida de la funcion y su derivada

            e = (d - yp)*dyp # Error obtenido entre la salida deseada y la obtenida
            self.w += ((eta/p)*np.dot(e, x.T)).T # Ajuste de los pesos sinapticos
            self.b += (eta/p)*np.sum(e, axis=1).reshape(-1,1) # Ajuste del bias

    def predict(self, x):
        v = np.dot(self.w.T, x) + self.b # Interaccion de la entrada con los pesos y el bias
        return self.function(v) # Salida de la funcion y su derivada

    def function(self, v, a=1):
        # Funcion escalon
        if self.f == 0: # step
            y = 1*(y>=0)
            dy = np.ones(v.shape)
            return y, dy

        # Funcion lineal o identidad
        if self.f == 1: # linear
            y = v
            dy = np.ones(v.shape)
            return y, dy

        # Funcion logistica
        if self.f == 2: # logistic
            y = 1/(1 + np.exp(-a*v))
            dy = a*y*(1 - y)
            return y, dy

        # Funcion softmax
        if self.f == 3: # softmax
            e = np.exp(v - np.max(v,axis=0))
            y = e/np.sum(e,axis=0)
            dy = np.ones(v.shape)
            return y, dy

