function [CM, dist] = findCMs_logvote(dI, alpha, dim)
    if nargin < 3
        dim = 1;
    end

    sz = size(dI);
    n = sz(dim);

    % Generate coordinate grid along the specified dimension
    idx = repmat({':'}, 1, ndims(dI));
    x0 = reshape(1:n, [ones(1,dim-1), n, ones(1,ndims(dI)-dim)]);
    mX = mean(x0(:));
    x = x0 - mX;

    % Exponential weights
    D = alpha * cumsum(dI, dim);
    D = D - mean(D, dim);
    epD = exp(D);

    % Prepare numerator and denominator for weighted average
    num = x .* epD;
    den = epD;

    % Define a window size for local averaging (can be tuned)
    win = 15; % Example window size
    kernel = exp(-abs((-win:win))); % Symmetric exponential kernel

    % Convolve numerator and denominator along the specified dimension
    num_conv = convn(num, reshape_kernel(kernel, dim, ndims(dI)), 'same');
    den_conv = convn(den, reshape_kernel(kernel, dim, ndims(dI)), 'same');

    CM = num_conv ./ den_conv + mX;
    CM(isinf(CM)) = nan;

    if nargout > 1
        dist = abs(CM - x0);
    end
end

function k = reshape_kernel(kernel, dim, nd)
    % Reshape kernel for convolution along specified dimension
    sz = ones(1, nd);
    sz(dim) = numel(kernel);
    k = reshape(kernel, sz);
end
