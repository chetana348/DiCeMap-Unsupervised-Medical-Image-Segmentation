function [CM, dist] = findCMs_logvote(dI, alpha, dim)
if nargin < 3, dim = 1; end
sz = size(dI);
reshapeArg = ones(1, ndims(dI)); reshapeArg(dim) = sz(dim);
x0 = reshape(1:size(dI, dim), reshapeArg); mX = mean(x0(:)); x = x0 - mX;

D = alpha * cumsum(dI, dim);
D = D - mean(D, dim);

% Introduce skew or sharpening through beta param
beta = 1.2;  % tune this as needed
epD = exp(beta * D);
enD = exp(-beta * D);

numer = enD .* cumsum(x .* epD, dim) + epD .* cumsum(x .* enD, dim, 'reverse');
denom = enD .* cumsum(epD, dim) + epD .* cumsum(enD, dim, 'reverse');
CM = numer ./ (denom + eps) + mX;
CM(isinf(CM) | isnan(CM)) = nan;

if nargout > 1, dist = abs(CM - x0); end
end
