%% Ian Mu;oz Nu;ez - RBF (Redes Neuronales de Base Radial)

close all
clear
clc

pkg load statistics

xl = 0; % Limite inferior de la funcion
xu = 2*pi; % Limite superior de la funcion
n = 20; % Numero de elementos

%% Patron de entrada
x = linspace(xl, xu, n);
y = linspace(xl, xu, n);
xy = [x; y];
[X, Y] = meshgrid(x, y);

%% Salida deseada
Z = cos(X) - 3*sin(Y);

figure(1)

plot3(X, Y, Z, 'r*', 'LineWidth', 6, 'MarkerSize', 4)

k = 12; % Numero de nucleos

[~, mu] = kmeans(xy', k); % Distribucion de los nucleos
mu = mu';

sigma = (max(max(mu))-min(min(mu)))/sqrt(2*k); % Desviacion estandar

G = zeros(n,k);
for i= 1:n
    for j= 1:k
        dist = norm(xy(:,i)-mu(:,j),2); % Distancia Euclidiana
        G(i,j) = exp(-(dist^2)/(2*(sigma^2))); % Funcion de Base Radial
    end
end

W = pinv(G) * Z'; % Pesos de la red

zp = G * W; % Prediccion de la red

figure(1)
hold on
grid on

surf(X, Y, zp')

title("Prediccion", 'FontSize', 20)
xlabel('x', 'FontSize', 15)
ylabel('y', 'FontSize', 15)
zlabel('z', 'FontSize', 15)

