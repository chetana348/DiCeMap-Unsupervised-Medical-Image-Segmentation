function [CM, dist] = findCMs_logvote(dI, alpha, dim)
% FINDCMS_REFACTORED - Refactored original method (behaviorally identical, algebraically distinct)

if nargin < 3
    dim = 1;
end

sz = size(dI);
reshapeArg = ones(1, ndims(dI));
reshapeArg(dim) = sz(dim);
x0 = reshape(1:size(dI, dim), reshapeArg);
mX = mean(x0(:));
x = x0 - mX;

% Directional accumulation
F = alpha * cumsum(dI, dim);
F = F - mean(F, dim);

% Compute symmetric exponentials
E = exp(F);
E_inv = 1 ./ E;  % instead of recomputing exp(-F)

% Apply refactored moment computation
xE = x .* E;
xEinv = x .* E_inv;

% Forward-backward weighted average using refactored notation
forward = E_inv .* cumsum(xE, dim);
backward = E .* cumsum(xEinv, dim, 'reverse');

normF = E_inv .* cumsum(E, dim);
normB = E .* cumsum(E_inv, dim, 'reverse');

CM = (forward + backward) ./ (normF + normB + eps) + mX;

% Cleanup
CM(isnan(CM) | isinf(CM)) = nan;

if nargout > 1
    dist = abs(CM - x0);
end
end
