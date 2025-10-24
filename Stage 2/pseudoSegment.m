function [L, distCM] = pseudoSegment(I, L)

existL = exist('L', 'var');
D = 2;

power = 2;
alpha = 2700;
maxIter = 500;
randIterNum = 300;
angleStep = 1;
s0 = size(I);
s0 = s0(1:D);
mSize = max(s0);
I = double(I);
outsideValue = I;
for d = 1:D
    outsideValue = max(outsideValue, [], d);
end
outsideValue = 0; %2 * outsideValue * crop; 
cI = bsxfun(@times, ones(mSize*ones(1,D)), outsideValue);
cropX = cell(1, D);
for d = 1:D
    cropStart = round((mSize - s0(d)) / 2);
    cropX{d} = cropStart + (1:s0(d));  
end
cI(cropX{:},:) = I;
clear I
angles = 0:angleStep:179;

sBox = single(size(cI));
sBox = sBox(1:D);
s = sBox;

sA = size(angles,2);
padZeros = zeros([1 double(sBox(2:end))]);
X = cell(D,1);
cellOne2S = cell(1, D);  % preallocate cell array
for d = 1:D
    cellOne2S{d} = 1:sBox(d);  % generate vector indices
end
[X{:}] = ndgrid(cellOne2S{:});
X = cellfun(@uint16, X, 'UniformOutput', false);

aX = zeros([sA s D], 'uint16');
aD = zeros([sA s]  , 'uint16');
pool = [];

cI = bsxfun(@minus, cI, outsideValue);
disp('Creating the CM images...')

tic
for a = 1:sA  
    disp(['Direction # ' num2str(a) ' / ' num2str(sA)])
    Irot_D = bsxfun(@plus, imrotate(cI, angles(a), 'bilinear', 'crop'), outsideValue);
    Irot_D = [abs(diff(Irot_D,1,1)).^power; padZeros];
    [C, Irot_D] = weightedMedian(Irot_D, alpha); % Irot_D is now the distance from the CM.
    C = round(single(C));
    C(C<1 | C>mSize) = nan;
    Irot_D = gather(uint16(round(imrotate(single(Irot_D), -angles(a), 'bilinear', 'crop'))));
    aD(a,:,:) = Irot_D;
    nanC = isnan(C(:)) | isnan(X{2}(:));
    C(nanC) = 1;
    C = sub2ind(sBox, C(:), X{2}(:)); % C is now the CM indices.
    C(nanC) = 1;
    for d = 1:D
        Xrot = imrotate(X{d}, angles(a), 'nearest', 'crop');
        Xrot = Xrot(C);
        Xrot(nanC | isnan(Xrot)) = 0;
        Xrot = gather(imrotate(reshape(Xrot, sBox), -angles(a), 'nearest', 'crop'));
        aX(a,:,:,d) = Xrot;
    end

end
t = toc;
disp(['CM images were created for ' num2str(sA) ' directions in ' num2str(t) ' seconds.'])

clear I Irot_D Xrot C nanC

mask = ~any(isnan(aX), D+2) & all(aX>=1, D+2) & all(bsxfun(@le, aX, permute(s, [3:(2+D), 1 2])), D+2);
aD = single(aD);
mask = mask & bsxfun(@times, rand([sA s], 'single'), max(aD, [], 1).^(D-1)) <= aD.^(D-1);
if nargout > 1
    if ~existL
        L = [];
    end
    aDsum = sum(aD .* ~isnan(aD), 1);
    aDcount = sum(~isnan(aD), 1);
    distCM = squeeze(aDsum ./ aDcount);
    return
end
delete(pool)
clear aD

clear cI

aX = num2cell(aX, 1:(D+1));
aX = cellfun(@(c) c(mask(:)), aX, 'UniformOutput', false);
indL = uint32(sub2ind(s, aX{:}));
clear aX
nMask = single(squeeze(sum(mask,1)));
indM = nMask(:)>0;
X = cellfun(@(c) c(indM), X, 'UniformOutput', false);
nMask = nMask(indM); sNMask = length(nMask);
mask = false([sA s]);
mask(sub2ind([sA s], nMask, X{:})) = true;
mask = cumsum(mask, 1, 'reverse')>0;

nIter = 1;
if existL
    L = single(L);
else
    L = nan(s, 'single');
end
r = setdiff(1:prod(s), unique(L(:)));
L(isnan(L(:))) = r(randperm(nnz(isnan(L(:)))));
clear r

disp('Random iterations...')
maskSize = find(~any(any(any(mask,2),3),4), 1, 'first');
if isempty(maskSize)
    sT = [sA s];
else
    sT = [maskSize-1, s];
    mask = mask(1:sT(1),:,:,:);
end
keepWhile = true;
while keepWhile && nIter<=maxIter
    verbPhrase = ['Iteration # ' num2str(nIter) ' out of ' num2str(maxIter) '.'];
    disp(verbPhrase)
    L0 = L;
    aL = nan(sT, 'like', L);
    aL(mask(:)) = L(indL);
    if nIter > randIterNum
        L = squeeze(mode(aL, 1));
    else
        L(indM) = aL(sub2ind(sT, round(rand(sNMask, 1, 'like', s) .* (nMask-1)) + 1, X{:}));
        if nIter == randIterNum
            disp('Random phase finished.')
        end
    end
    nanL = isnan(L);
    L(nanL) = L0(nanL);
    if isequaln(L,L0)
        keepWhile = false;
    end
    [~, ~, L(:)] = unique(L(:));

    nIter = nIter + 1;
end
disp(['Segmentation took ' num2str(toc) ' seconds.'])

end

