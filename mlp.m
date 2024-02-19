% Ian Mu;oz Nu;ez - MLP (Perceptron Multicapa)

function [model, loss] = mlp(x, y, k, eta, epocas)
    loss = inf(1, epocas); % Arreglo para la funcion de perdida de la red

    k = [size(x,1); k(:); size(y,1)]; % Numero total de capas
    L = numel(k)-1; % Numero de capas ocultas
    w = cell(L,1); % Arreglo para los pesos sinapticos de las capas
    b = cell(L,1); % Arreglo para los bias de las capas

    xl = -1; xu = 1;
    for l= 1:L
        w{l} = xl + (xu-xl)*rand(k(l), k(l+1)); % Se generan los pesos sinapticos para cada capa aleatoriamente
        b{l} = xl + (xu-xl)*rand(k(l+1), 1); % Se generan los bias para cada capa aleatoriamente
    end

    delta = cell(L,1); % Arreglo para los errores de las salidas de las capas
    phi = cell(L+1,1); % Arreglo para las entradas/salidas de las capas
    phi{1} = x; % Entrada de la red

    p = size(x,2);
    for epoca= 1:epocas
        % Etapa hacia adelante
        for l= 1:L-1
            v = (w{l}'*phi{l}) + b{l};
            phi{l+1} = tanh(v);
        end
        v = (w{L}'*phi{L}) + b{L};
        e = exp(v - max(v));
        phi{L+1} = e./sum(e,1);

        % Etapa hacia atras
        delta{L} = (y - phi{L+1}).*ones(size(v));
        loss(epoca) = dot(delta{L}(:), delta{L}(:));
        for l= L-1:-1:1
            df = (1-phi{l+1}).*(1+phi{l+1});
            delta{l} = (w{l+1}*delta{l+1}).*df;
        end

        % Ajuste de pesos y bias
        for l= 1:L
            w{l} = w{l} + (eta/p)*(phi{l}*delta{l}');
            b{l} = b{l} + (eta/p)*sum(delta{l},2);
        end
    end

    model.w = w;
    model.b = b;
end

