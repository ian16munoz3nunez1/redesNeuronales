#!/bin/python3

# Ian Mu;oz Nu;ez - MLP (Perceptron Multicapa)

import numpy as np
from colorama import init
from colorama.ansi import Fore

init(autoreset=True)

class MLP:
    def __init__(self, k, scaler=False):
        self.scaler = scaler
        functions = ['step', 'linear', 'relu', 'tanh', 'logistic', 'softmax'] # Funciones de activacion para la red
        self.L = len(k)-1 # Numero de capas ocultas

        self.w = {} # Arreglo para los pesos sinapticos de las capas
        self.b = {} # Arreglo para los bias de las capas
        self.f = {} # Arreglo para las funciones de activacion
        self.phi = {} # Arreglo para las entradas/salidas de las capas
        self.dphi = {} # Arreglo para la derivada de las salidas de las capas
        self.delta = {} # Arreglo para los errores de las salidas de las capas

        xl, xu = -1, 1
        for l in range(1,self.L+1):
            self.w[l] = xl + (xu-xl)*np.random.rand(k[l-1][0], k[l][0]) # Se generan los pesos sinapticos para cada capa aleatoriamente
            self.b[l] = xl + (xu-xl)*np.random.rand(k[l][0],1) # Se generan los bias para cada capa aleatoriamente
            if k[l][1] in functions:
                self.f[l] = k[l][1] # Se generan las funciones de activacion para las capas
            else:
                print(Fore.RED + f"Function \"{k[l][1]}\" not available")
                exit(1)

    # Funcion para entrenar la red
    def fit(self, x, y, eta, epocas, batch_size=None):
        if self.scaler:
            x, self.ux, self.sx = self.scaled(x)
            y, self.uy, self.sy = self.scaled(y)

        loss = np.zeros(epocas) # Arreglo para la funcion de perdida de la red
        p = x.shape[1] # Numero de patrones de entrada
        self.phi[0] = x # Entrada de la red

        for l in range(1, self.L+1):
            self.phi[l] = np.zeros((self.b[l].shape[0], p))
            self.dphi[l] = np.zeros((self.b[l].shape[0], p))
            self.delta[l] = np.zeros((self.phi[l].shape[0], self.phi[l].shape[1]))

        if batch_size is None:
            batch_size = p

        for epoca in range(epocas):
            xl, xu = 0, batch_size
            while xl < p:
                # Etapa hacia adelante
                for l in range(1, self.L+1):
                    v = np.matmul(self.w[l].T, self.phi[l-1][:,xl:xu]) + self.b[l]
                    self.phi[l][:,xl:xu], self.dphi[l][:,xl:xu] = self.function(self.f[l], v)
                    l += 1

                # Etapa hacia atras
                self.delta[self.L][:,xl:xu] = (y[:,xl:xu] - self.phi[self.L][:,xl:xu])*self.dphi[self.L][:,xl:xu]
                l = self.L-1
                while l > 0:
                    self.delta[l][:,xl:xu] = np.matmul(self.w[l+1], self.delta[l+1][:,xl:xu])*self.dphi[l][:,xl:xu]
                    l -= 1

                # Ajuste de pesos y bias
                for l in range(1, self.L+1):
                    self.w[l] += (eta/p)*np.matmul(self.phi[l-1][:,xl:xu], self.delta[l][:,xl:xu].T)
                    self.b[l] += (eta/p)*np.sum(self.delta[l][:,xl:xu],1).reshape(-1,1)

                xl += batch_size
                xu += batch_size

            loss[epoca] = np.dot(self.delta[self.L].flatten(), self.delta[self.L].flatten())

        return loss

    # Funcion para regresar la prediccion de la red
    def predict(self, x):
        x = (x-self.ux)/self.sx if self.scaler else x

        phi = np.array(x)
        for l in range(1, self.L+1):
            v = np.matmul(self.w[l].T, phi) + self.b[l]
            phi = self.function(self.f[l], v)[0]

        phi = (phi*self.sy)+self.uy if self.scaler else phi
        return phi

    # Funciones de activacion y su derivada
    def function(self, f, v, a=1, b=1):
        if f == 'step':
            y = 1*(v>=0)
            dy = np.ones(v.shape)
            return y, dy

        elif f == 'linear':
            y = v
            dy = np.ones(v.shape)
            return y, dy

        elif f == 'relu':
            y = v*(v>=0)
            dy = np.array(v>=0, dtype=np.float64)
            return y, dy

        elif f == 'tanh':
            y = a*np.tanh(b*v)
            dy = (b/a)*(1-y)*(1+y)
            return y, dy

        elif f == 'logistic':
            y = 1/(1 + np.exp(-a*v))
            dy = a*y*(1-y)
            return y, dy

        elif f == 'softmax':
            e = np.exp(v - np.max(v,axis=0))
            y = e/np.sum(e,axis=0)
            dy = np.ones(v.shape)
            return y, dy

        else:
            print(Fore.RED + f"Function \"{f}\" not available")

    # 'score' regresa la probabilidad del buen entrenamiento de la red
    def score(self, x, y):
        yp = self.predict(x)
        return self.metrics(y, yp, 'R2')

    # Funcion para regresar el muestras para entrenamiento y generalizacion
    def train_test_split(self, x, y, train_size):
        if train_size > 1:
            print(Fore.YELLOW + "[!] Train size must be less than one")
            exit(1)

        p = x.shape[1]
        n = int(np.floor(p*train_size))
        permutation = np.random.permutation(p)
        xTrain, yTrain = x[:,permutation[:n]], y[:,permutation[:n]]
        xTest, yTest = x[:,permutation[n:]], y[:,permutation[n:]]

        return xTrain, xTest, yTrain, yTest

    def scaled(self, x):
        u = x.mean()
        s = np.sqrt(np.sum((x-u)**2)/x.size)
        x = (x-u)/s
        return x, u, s

    # Metricas de regresion
    def metrics(self, y, yp, error):
        n = y.shape[1]
        if error.upper() == 'MAE':
            return np.mean(np.abs(y - yp))

        elif error.upper() == 'MSE':
            return np.mean((y-yp)**2)

        elif error.upper() == 'MEDAE':
            return np.median(np.abs(y-yp))

        elif error.upper() == 'R2':
            mean = np.mean(y)
            return 1 - (np.sum((y-yp)**2)/np.sum((y-mean)**2))

        elif error.upper() == 'EVS':
            return 1 - (np.var(y-yp)/np.var(y))

        else:
            print(Fore.RED + f"Unrecognized metric \'{error}\'")

