% Ian Mu;oz Nu;ez - MLP (Perceptron Multicapa)

function yp = mlpPred(model, x)
    w = model.w;
    b = model.b;
    L = length(w);
    yp = x;

    for l= 1:L-1
        yp = tanh((w{l}'*yp) + b{l});
    end
    v = (w{L}'*yp) + b{L};
    e = exp(v - max(v));
    yp = e./sum(e,1);
end
