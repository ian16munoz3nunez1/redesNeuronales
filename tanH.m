% Ian Mu;oz Nu;ez - Funcion 'Tangente Hiperbolica'

close all
clear
clc

v = linspace(-10, 10, 1000); % Variable independiente

a = 1; % Parametro que determina la magnitud de la funcion
b = 1; % Parametro que determina la frecuencia de la funcion
c = 0; % Parametro que determina la fase de la funcion
y = a*tanh(b*v + c); % Funcion o variable dependiente
dy = (b/a).*(1-y).*(1+y); % Derivada de la funcion

figure(1)

subplot(2,1,1)
hold on
grid on

plot(v, y, 'g-', 'LineWidth', 2)
title("Funcion tangente hiperbolica", 'FontSize', 20)
xlabel('v', 'FontSize', 15)
ylabel('\phi(v)', 'FontSize', 15)

subplot(2,1,2)
hold on
grid on

plot(v, dy, 'r-', 'LineWidth', 2)
title("Derivada de la funcion", 'FontSize', 20)
xlabel('v', 'FontSize', 15)
ylabel('d \phi(v)', 'FontSize', 15)

