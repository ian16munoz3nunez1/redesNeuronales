#!/bin/python3

# Ian Mu;oz Nu;ez - Adaline
# Desarrollar un código en el que un Adaline sea capaz de aproximar una
# función generada con datos aleatorios utilizando dos tipos de métodos,
# el SGD (Stochastic Gradient Descent) y BDG (Batch Gradient Descent)
# para comparar su funcionamiento
#############
#### SGD ####
#############
# Realiza un ajuste de pesos y bias con cada uno de los patrones de
# entrada, al hacer esto, se ajusta en pocas epocas, pero con muchos
# patrones de entrada se vuelve lento y tarda más tiempo en terminar
# el entrenamiento y alcanzar un ajuste óptimo.
#############
#### BGD ####
#############
# Este método realiza el ajuste de pesos y bias utilizando todos los
# patrones de entrada, debido a esto, el ajuste optimo tarda más
# epocas en ser alcanzado, pero al haber muchos patrones de entrada
# no afecta mucho al tiempo de entrenamiento ni a los recursos
# computacionales

import numpy as np

class Adaline:
    # Funcion para entrenar el Adaline
    def fit(self, x, y, eta, epocas, solver):
        self.w = np.random.rand() # Pesos sinapticos
        self.b = np.random.rand() # Bias

        p = x.shape[0] # Numero de patrones de entrada
        J = np.zeros(epocas) # Arreglo para almacenar el error del Adaline

        if solver == 'SGD': # Stocastic Gradient Descent Method
            for epoca in range(epocas):
                for i in range(p):
                    yp = np.dot(self.w, x[i]) + self.b # Interaccion de la entrada con los pesos y el bias

                    e = y[i] - yp # Error entre la salida deseada y la obtenida
                    self.w += eta*e*x[i] # Ajuste de los pesos sinapticos
                    self.b += eta*e # Ajuste del valor del bias

                J[epoca] = np.sum((y - (np.dot(self.w,x) + self.b))**2) # Error minimo cuadrado

        if solver == 'BGD': # Batch Gradient Descent Method
            for epoca in range(epocas):
                yp = np.dot(self.w, x) + self.b # Interaccion de la entrada con los pesos y el bias

                e = y - yp # Error entre la salida deseada y la obtenida
                self.w += (eta/p) * np.dot(e, x) # Ajuste de los pesos sinapticos
                self.b += (eta/p) * np.sum(e) # Ajuste del valor del bias

                J[epoca] = np.sum((y - (np.dot(self.w,x) + self.b))**2) # Error minimo cuadrado

        return J

    # Funcin para generar una salida con un patron de entrada
    def predict(self, x):
        p = x.shape[0] # Numero de patrones de entrada
        yp = np.zeros(p) # Salida obtenida

        for i in range(p):
            yp[i] = np.dot(self.w, x[i]) + self.b # Interaccion de la entrada con los pesos y el bias

        return yp

