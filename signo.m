% Ian Mu;oz Nu;ez - Funcion 'Signo'

close all
clear
clc

v = linspace(-10, 10, 1000); % Variable independiente

y = []; % Funcion o variable dependiente
for i= 1:1000
    if v(i) > 0
        y(i) = 1;
    elseif v(i) == 0
        y(i) = 0;
    else
        y(i) = -1;
    end
end
dy = zeros(1,1000); % Derivada de la funcion

figure(1)

subplot(2,1,1)
hold on
grid on

plot(v, y, 'g-', 'LineWidth', 2)
title("Funcion signo", 'FontSize', 20)
xlabel('v', 'FontSize', 15)
ylabel('\phi(v)', 'FontSize', 15)

subplot(2,1,2)
hold on
grid on

plot(v, dy, 'r-', 'LineWidth', 2)
title("Derivada de la funcion", 'FontSize', 20)
xlabel('v', 'FontSize', 15)
ylabel('d \phi(v)', 'FontSize', 15)

