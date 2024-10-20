#!python3

import numpy as np
from sklearn.cluster import KMeans

class RBF():
    def __init__(self, k=1):
        self.__k = k # Numero de nucleos

    def fit(self, x, y, n):
        k = self.__k # Numero de nucleos
        model = KMeans(n_clusters=k) # Modelo de KMeans
        model.fit(x) # Ajuste del modelo KMeans

        mu = model.cluster_centers_ # Distribucion de los nucleos
        sigma = (np.max(mu)-np.min(mu))/np.sqrt(2*k) # Desviacion estandar
        self.__mu = mu
        self.__sigma = sigma

        G = np.zeros((n,k)) # Matriz de funciones
        for i in range(n):
            for j in range(k):
                dist = np.linalg.norm(x[i,0]-mu[j],2) # Distancia euclidiana
                G[i,j] = np.exp(-(dist**2)/(sigma**2)) # Funcion de base radial

        self.__W = np.dot(np.linalg.pinv(G), y) # Pesos de la red

    def predict(self, x):
        n = x.shape[0] # Numero de patrones de x
        k = self.__k # Numero de nucleos

        mu = self.__mu # Distribucion de los nucleos
        sigma = self.__sigma # Desviacion estandar
        W = self.__W # Pesos de la red

        G = np.zeros((n,k)) # Matriz de funciones
        for i in range(n):
            for j in range(k):
                dist = np.linalg.norm(x[i,0]-mu[j],2) # Distancia euclidiana
                G[i,j] = np.exp((-dist**2)/(sigma**2)) # Funcion de base radial

        y = np.dot(G, W) # Prediccion de la red

        return y

