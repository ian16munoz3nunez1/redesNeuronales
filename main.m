% Ian Mu;oz Nu;ez - Perceptron Unicapa
% Desarrollar un codigo en el que un perceptron unicapa sea
% capaz de clasificar 4 grupos utilizando la funcion
% logistica y el esquema One-vs-All
%           1
% y = -------------
%     1 + exp(-a*v)

seed = [0 0 1 1;
        0 1 0 1]; % Puntos de referencia para los puntos
clases = 4; % Numero de clases deseadas
p = 20; % Numero de patrones por clase
n_inputs = 2; % Numero de entradas de la red
n_outputs = 4; % Numero de salidas de la red

% Llenado de los patrones de entrada y la salida deseada
for i= 1:clases
    x(:, (p*i)-(p-1):p*i) = seed(:,i) + 0.15*rand(n_inputs,p);
    d(i, (p*i)-(p-1):p*i) = ones(1,p);
end

a = 1; % Parametro para modificar el comportamiento de una funcion
eta = 0.2; % Factor de aprendizaje de la red
epocas = 10000; % Numero de iteraciones deseadas
xl = -2; % Limite inferior para los pesos y bias
xu = 2; % Limite superior para los pesos y bias
w = xl + (xu-xl)*rand(n_inputs,n_outputs); % Pesos sinapticos
b = xl + (xu-xl)*rand(n_outputs,1); % Bias
p = size(x,2); % Numero de patrones de entrada

for epoca= 1:epocas
    v = w' * x + b; % Interaccion de la entrada con los pesos y el bias
    y = 1./(1 + exp(-a*v)); % Salida de la funcion
    dy = a*y.*(1 - y); % Derivada de la funcion

    e = (d - y).*dy; % Error obtenido entre la salida deseada y la obtenida
    w = w + ((eta/p)*(e*x'))'; % Ajuste de los pesos sinapticos
    b = b + (eta/p)*sum(e,2); % Ajuste del bias
end

xl = min(min(x))-0.1; % Limite inferior para los hiperplanos
xu = max(max(x))+0.1; % Limite superior para los hiperplanos
t = xl:0.1:xu; % Arreglo de valores para los hiperplanos
colors = [1 0 0; 0 1 0; 0 0 1; 1 1 0]; % Colores para diferenciar los grupos

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

for i= 1:clases
    f = (-w(1,i)/w(2,i))*t - (b(i,1)/w(2,i)); % Funcion del hiperplano separador
    plot(t, f, '-', 'Color', colors(i,:), 'LineWidth', 2) % Grafica de los hiperplanos
end

title("OLP", 'FontSize', 20)
xlabel('x', 'FontSize', 15)
ylabel('y', 'FontSize', 15)

