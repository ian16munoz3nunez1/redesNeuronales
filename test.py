#!/bin/python3

# Ian Mu;oz Nu;ez - Perceptron
# - Construya la tabla de verdad para la siguiente función
#   lógica: A(B+C)
# - Escriba un código para implementar la función
#       +-------------------+
#       | A | B | C | F | b |
#       |---+---+---+---+---|
#       | 0 | 0 | 0 | 0 | 1 | <-- Patron 1
#       |---+---+---+---+---|
#       | 0 | 0 | 1 | 0 | 1 | <-- Patron 2
#       |---+---+---+---+---|
#       | 0 | 1 | 0 | 0 | 1 | <-- Patron 3
#       |---+---+---+---+---|
#       | 0 | 1 | 1 | 0 | 1 | <-- Patron 4
#       |---+---+---+---+---|
#       | 1 | 0 | 0 | 0 | 1 | <-- Patron 5
#       |---+---+---+---+---|
#       | 1 | 0 | 1 | 1 | 1 | <-- Patron 6
#       |---+---+---+---+---|
#       | 1 | 1 | 0 | 1 | 1 | <-- Patron 7
#       |---+---+---+---+---|
#       | 1 | 1 | 1 | 1 | 1 | <-- Patron 8
#       +-------------------+

# Importacion de modulos
import numpy as np
import matplotlib.pyplot as plt
import pickle

nn = pickle.load(open('perceptron.sav', 'rb')) # Se cargan los datos del archivo sav

xl, xu = -0.1, 1.1 # Limites para los datos aleatorios de entrada
x = xl + (xu-xl)*np.random.rand(3, 20) # Datos de entrada

yp = nn.predict(x) # Se obtiene la salida de la red con el entrenamiento previo

w1, w2, w3, b = nn.w[0], nn.w[1], nn.w[2], nn.b # Pesos y bias del perceptron
xLim = np.linspace(xl, xu, 100) # Valores en X para generar la malla
yLim = np.linspace(xl, xu, 100) # Valores en Y para generar la malla
X, Y = np.meshgrid(xLim, yLim) # Malla con rangos de valores de xl a xu para el hiperplano separador

m = -((w1*X)/w3) - ((w2*Y)/w3) # Pendiente del hiperplano
b = -b/w3 # Coeficiente del hiperplano
f = m + b # Funcion del hiperplano separador
# f = -((w1*X)/w3) - ((w2*Y)/w3) - (b/w3) # Funcion del hiperplano separador

plt.figure(1)
ax = plt.axes(projection='3d')
ax.axis('equal') # Muestra la escala de los ejes igual
ax.set_xlim([xl, xu]) # Limites del eje 'x'
ax.set_ylim([xl, xu]) # Limites del eje 'y'
ax.set_zlim([xl, xu]) # Limites del eje 'z'

# Grafica de los datos de entrenamiento y su clasificacion
ax.plot3D(x[0,yp==1], x[1,yp==1], x[2,yp==1], 'yo', markersize=8)
ax.plot3D(x[0,yp==0], x[1,yp==0], x[2,yp==0], 'go', markersize=8)
ax.plot_surface(X, Y, f, cmap='viridis') # Grafica del hiperplano separador

# Informacion de la grafica
ax.set_title("A(B+C)", fontsize=20)
ax.set_xlabel('A', fontsize=15)
ax.set_ylabel('B', fontsize=15)
ax.set_zlabel('C', fontsize=15)

plt.show()

