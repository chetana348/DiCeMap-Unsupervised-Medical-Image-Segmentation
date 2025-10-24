function [CM, dist] = weightedMedian_logvote(dI, alpha, dim)
    if nargin < 3
        dim = 1;
    end
    sz = size(dI);
    x = reshape(1:sz(dim), [], 1);
    mX = mean(x);
    x = x - mX;

    signal = alpha * cumsum(dI, dim);
    signal = signal - mean(signal, dim);

    weights = abs(signal);
    num = cumsum(bsxfun(@times, x, weights), dim);
    denom = cumsum(weights, dim);

    CM = num ./ (denom + 1e-8) + mX;

    if nargout > 1
        x0 = reshape(1:sz(dim), [], 1);
        dist = abs(CM - x0);
    end
end
