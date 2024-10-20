% Ian Mu;oz Nu;ez

close all
clear
clc

pkg load statistics

xl = -5; % Limite inferior de la funcion
xu = 5; % Limite superior de la funcion
n = 10; % Numero de elementos de entrada

x = linspace(xl, xu, n)'; % Datos de entrada
y = sin(x); % Salida deseada

figure(1)
hold on
grid on
plot(x, y, 'r*', 'LineWidth', 2, 'MarkerSize', 8)

k = 8; % Numero de nucleos
[~, mu] = kmeans(x,k); % Distribucion de los nucleos

sigma = (max(mu)-min(mu))/sqrt(2*k); % Desviacion estandar

G = zeros(n,k); % Matriz de funciones
for i=1:n
    for j=1:k
        dist = norm((x(i)-mu(j)), 2); % Distancia euclidiana
        G(i,j) = exp(-(dist^2)/(2*(sigma^2))); % Funcion de base radial
    end
end

W = pinv(G) * y; % Pesos de la red

n = 200; % Numero de muestras nuevas
x = linspace(xl, xu, n); % Datos de entrada

G = zeros(n,k); % Matriz para las funciones de base radial
for i=1:n
    for j=1:k
        dist = norm((x(i)-mu(j)), 2); % Distancia euclidiana
        G(i,j) = exp(-(dist^2)/(2*(sigma^2))); % Funcion de base radial
    end
end

yp = G * W; % Prediccion de la red

figure(1)
hold on
grid on

plot(x, yp, 'b-', 'LineWidth', 2)

title("Funcion seno", 'FontSize', 20)
xlabel('x', 'FontSize', 15)
ylabel('y', 'FontSize', 15)

