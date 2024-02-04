% Ian Mu;oz Nu;ez - Perceptron Unicapa
% Desarrollar un codigo en el que un perceptron unicapa sea
% capaz de clasificar multiples grupos utilizando la funcion
% softmax
%            z_j
%           e
% y_j = ----------
%         m
%        ---
%        \    z_i
%        /   e
%        ---
%        i=1

close all
clear
clc

clases = 8; % Numero de clases deseadas
p = 80; % Numero de patrones por clase
n_inputs = 2; % Numero de entradas de la red
n_outputs = clases; % Numero de salidas de la red

% Llenado de los patrones de entrada y la salida deseada
for i= 1:clases
    seed = rand(2,1);
    x(:, (p*i)-(p-1):p*i) = seed + 0.15*rand(n_inputs,p);
    d(i, (p*i)-(p-1):p*i) = ones(1,p);
end

a = 1; % Parametro para modificar el comportamiento de una funcion
eta = 1; % Factor de aprendizaje de la red
epocas = 1000; % Numero de iteraciones deseadas
xl = -2; % Limite inferior para los pesos y bias
xu = 2; % Limite superior para los pesos y bias
w = xl + (xu-xl)*rand(n_inputs,n_outputs); % Pesos sinapticos
b = xl + (xu-xl)*rand(n_outputs,1); % Bias
p = size(x,2); % Numero de patrones de entrada

for epoca= 1:epocas
    v = w' * x + b; % Interaccion de la entrada con los pesos y el bias
    y = exp(v-max(v,[],1))./(sum(exp(v-max(v,[],1)),1)); % Salida de la funcion
    dy = ones(size(v)); % Derivada de la funcion

    e = (d - y).*dy; % Error obtenido entre la salida deseada y la obtenida
    w = w + ((eta/p)*(e*x'))'; % Ajuste de los pesos sinapticos
    b = b + (eta/p)*sum(e,2); % Ajuste del bias
end

colors = [1 0 0; 0 1 0; 0 0 1; 0 1 1; 1 0 1; 1 1 0; 0 0 0; 0.5 0.5 0.5]; % Colores para diferenciar los grupos

figure(1)

subplot(121)
hold on
grid on

[_, yc] = max(d);
for i= 1:size(x,2)
    plot(x(1,i), x(2,i), 'o', 'Color', colors(yc(i),:), 'LineWidth', 8, 'MarkerSize', 8) % Grafica de los patrones de entrada
end

title("Problema original", 'FontSize', 20)
xlabel('x', 'FontSize', 15)
ylabel('y', 'FontSize', 15)

subplot(122)
hold on
grid on

[_, yc] = max(y);
for i= 1:size(x,2)
    plot(x(1,i), x(2,i), 'o', 'Color', colors(yc(i),:), 'LineWidth', 8, 'MarkerSize', 8) % Grafica de los grupos clasificados
end

title("OLP", 'FontSize', 20)
xlabel('x', 'FontSize', 15)
ylabel('y', 'FontSize', 15)

