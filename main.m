%% Ian Mu;oz Nu;ez - RBF (Redes Neuronales de Base Radial)

close all
clear
clc

pkg load statistics

xl = 0; % Limite inferior de la funcion
xu = 1; % Limite superior de la funcion
n = 500; % Numero de elementos en el patron de entrada

x1 = xl + (xu-xl) * rand(1,n);
x2 = xl + (xu-xl) * rand(1,n);
x = [x1; x2]; % Patron de entrada
y = (1.3356 .* (1.5 .* (1 - x1)) + (exp(2 .* x1 - 1) .* sin(3 .* pi .* (x1 - 0.6).^2)) + (exp(3 .* (x2 - 0.5)) .* sin(4 .* pi .* (x2 - 0.9).^ 2))); % Salida deseada

k = 20; % Numero de nucleos

[~, mu] = kmeans(x', k); % Distribucion de los nucleos
mu = mu';

sigma = (max(max(mu))-min(min(mu)))/sqrt(2*k); % Desviacion estandar

G = zeros(n,k); % Matriz para la funcion de base radial
for i= 1:k
    for j= 1:n
        dist = norm(x(:,j)-mu(:,i), 2); % Distancia Euclidiana
        G(j,i) = exp(-(dist^2)/(2*(sigma^2))); % Funcion de Base Radial
    end
end

W = pinv(G) * y'; % Calculo de los pesos de la red

G = zeros(n,k); % Matriz para la funcion de base radial
for i= 1:k
    for j= 1:n
        dist = norm(x(:,j)-mu(:,i), 2); % Distancia Euclidiana
        G(j,i) = exp(-(dist^2)/(2*(sigma^2))); % Funcion de Base Radial
    end
end

yp = G * W; % Prediccion de la red

figure(1)
hold on
grid on

plot(y, 'b-', 'LineWidth', 2)
plot(yp, 'r-', 'LineWidth', 2)

title("Random function", 'FontSize', 20)
xlabel('x', 'FontSize', 15)
ylabel('y', 'FontSize', 15)
legend('function', 'prediction')

