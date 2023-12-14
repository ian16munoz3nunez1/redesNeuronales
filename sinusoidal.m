% Ian Mu;oz Nu;ez - Funcion 'Sinusoidal'

close all
clear
clc

v = linspace(-10, 10, 1000); % Variable independiente

A = 1; % Parametro que determina la magnitud de la funcion
B = 1; % Parametro que determina la frecuencia de la funcion
C = 0; % Parametro que determina la fase de la funcion
phi = A*sin(B*v + C); % Funcion o variable dependiente

figure(1)
hold on
grid on

plot(v, phi, 'g-', 'LineWidth', 2)

title("Funcion sinusoidal", 'FontSize', 20)
xlabel('v', 'FontSize', 15)
ylabel('\phi(v)', 'FontSize', 15)

