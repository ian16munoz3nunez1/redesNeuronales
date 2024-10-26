%% Ian Mu;oz Nu;ez - RBF (Redes Neuronales de Base Radial)

close all
clear
clc

pkg load statistics

xl = -5; % Limite inferior de la funcion
xu = 5; % Limite superior de la funcion
n = 20; % Numero de elementos

x = linspace(xl, xu, n); % Patron de entrada
y = sin(x); % Salida deseada

figure(1)
hold on
grid on
plot(x, y, 'r*', 'LineWidth', 6, 'MarkerSize', 4)

k = 8; % Numero de nucleos

[~, mu] = kmeans(x', k); % Distribucion de los nucleos
mu = mu';

sigma = (max(max(mu))-min(min(mu)))/sqrt(2*k); % Desviacion estandar

G = zeros(n,k);
for i= 1:k
    for j= 1:n
        dist = norm(x(:,j)-mu(:,i),2); % Distancia Euclidiana
        G(j,i) = exp(-(dist^2)/(2*(sigma^2))); % Funcion de Base Radial
    end
end

W = pinv(G) * y'; % Pesos de la red

n = 200; % Numero de elementos
x = linspace(xl, xu, n); % Patron de entrada

G = zeros(n,k);
for i= 1:k
    for j= 1:n
        dist = norm(x(:,j)-mu(:,i),2); % Distancia Euclidiana
        G(j,i) = exp(-(dist^2)/(2*(sigma^2))); % Funcion de Base Radial
    end
end

yp = G * W; % Prediccion de la red

figure(1)
hold on
grid on

plot(x, yp, 'b-', 'LineWidth', 2)

title("Funcion Seno", 'FontSize', 20)
xlabel('x', 'FontSize', 15)
ylabel('y', 'FontSize', 15)

