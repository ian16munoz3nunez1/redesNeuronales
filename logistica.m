% Ian Mu;oz Nu;ez - Funcion 'Logistica'

close all
clear
clc

v = linspace(-10, 10, 1000); % Variable independiente

a = 1; % Parametro que determina la pendiente de la funcion sigmoidal
y = 1./(1 + exp(-a*v)); % Funcion o variable dependiente
dy = a*y.*(1-y); % Derivada de la funcion

figure(1)

subplot(2,1,1)
hold on
grid on

plot(v, y, 'g-', 'LineWidth', 2)
title("Funcion logistica", 'FontSize', 20)
xlabel('v', 'FontSize', 15)
ylabel('\phi(v)', 'FontSize', 15)

subplot(2,1,2)
hold on
grid on

plot(v, dy, 'r-', 'LineWidth', 2)
title("Derivada de la funcion", 'FontSize', 20)
xlabel('v', 'FontSize', 15)
ylabel('d \phi(v)', 'FontSize', 15)

