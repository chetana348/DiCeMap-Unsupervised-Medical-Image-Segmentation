function [centerMap, offset] = weightedMedian(data, scale, axis)
    % Default to dimension 1 if not specified
    if ~exist('axis', 'var')
        axis = 1;
    end

    % Setup reshaping dimensions
    dimShape = ones(1, ndims(data));
    dimShape(axis) = size(data, axis);
    
    % Generate coordinate system for given axis
    coords = reshape(1:size(data, axis), dimShape);
    coordMean = mean(coords);
    centeredCoords = coords - coordMean;

    % Compute directional weight
    weightedSum = scale * cumsum(data, axis);
    clear data

    % Normalize weights
    weightedSum = weightedSum - mean(weightedSum, axis);

    % Exponentials of the weights
    expPos = exp(weightedSum);
    expNeg = exp(-weightedSum);
    clear weightedSum

    % Numerator and denominator for center of mass computation
    top = expNeg .* cumsum(centeredCoords .* expPos, axis) + ...
          expPos .* cumsum(centeredCoords .* expNeg, axis, 'reverse');
      
    bottom = expNeg .* cumsum(expPos, axis) + ...
             expPos .* cumsum(expNeg, axis, 'reverse');

    % Final center of mass along the specified axis
    centerMap = top ./ bottom + coordMean;
    clear expPos expNeg

    % Handle infinities
    centerMap(isinf(centerMap)) = NaN;

    % Optional output: offset from original coordinates
    if nargout > 1
        offset = abs(centerMap - coords);
    end
end
