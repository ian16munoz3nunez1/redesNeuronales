% Ian Mu;oz Nu;ez - Funcion 'Lineal a Tramos'

close all
clear
clc

v = linspace(-10, 10, 1000); % Variable independiente

y = []; % Funcion o variable dependiente
dy = []; % Derivada de la funcion
for i= 1:1000
    if v(i) >= 1
        y(i) = 1;
        dy(i) = 0;
    elseif v(i) > 0 && v(i) < 1
        y(i) = v(i);
        dy(i) = 1;
    else
        y(i) = 0;
        dy(i) = 0;
    end
end

figure(1)

subplot(2,1,1)
hold on
grid on

plot(v, y, 'g-', 'LineWidth', 2)
title("Funcion lineal a tramos", 'FontSize', 20)
xlabel('v', 'FontSize', 15)
ylabel('\phi(v)', 'FontSize', 15)

subplot(2,1,2)
hold on
grid on

plot(v, dy, 'r-', 'LineWidth', 2)
title("Derivada de la funcion", 'FontSize', 20)
xlabel('v', 'FontSize', 15)
ylabel('d \phi(v)', 'FontSize', 15)

