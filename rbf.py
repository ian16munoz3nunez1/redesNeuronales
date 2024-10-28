#!python3

# Ian Mu;oz Nu;ez - RBF (Redes Neuronales de Base Radial)

import numpy as np
from sklearn.cluster import KMeans

class RBF:
    def __init__(self, k=1):
        self.__k = k # Numero de nucleos
    
    def fit(self, x, y, n):
        k = self.__k # Numero de nucleos
        model = KMeans(n_clusters=k) # Modelo de KMeans
        model.fit(x.T) # Ajuste de la entrada al modelo KMeans

        mu = model.cluster_centers_ # Distribucion de los nucleos
        mu = mu.T
        sigma = (np.max(mu)-np.min(mu))/np.sqrt(2*k) # Desviacion estandar

        self.__mu = mu
        self.__sigma = sigma

        G = np.zeros((n,k))
        for i in range(k):
            for j in range(n):
                dist = np.linalg.norm(x[:,j]-mu[:,i],2) # Distancia Euclidiana
                G[j,i] = np.exp(-(dist**2)/(2*(sigma**2))) # Funcion de Base Radial

        self.__W = np.dot(np.linalg.pinv(G), y.T) # Pesos de la red

    def predict(self, x):
        n = x.shape[1] # Numero de patrones de 'x'
        k = self.__k # Numero de nucleos
        mu = self.__mu # Distribucion de los nucleos
        sigma = self.__sigma # Desviacion estandar
        W = self.__W # Pesos de la red

        G = np.zeros((n,k))
        for i in range(k):
            for j in range(n):
                dist = np.linalg.norm(x[:,j]-mu[:,i],2) # Distancia Euclidiana
                G[j,i] = np.exp(-(dist**2)/(2*(sigma**2))) # Funcion de Base Radial

        y = np.dot(G, W) # Prediccion de la red

        return y

