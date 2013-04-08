function [indextab valstab coeffstab] = staticsPrecompute(indices, values, coeffs, arrdims)
% Given a list of indices (x,y,z), the array dimensions, and associated values and coeffs,
% compute all possible permutations

if (numel(values) == 0) || (numel(coeffs) == 0) || (numel(indices) == 0);
    valstab = [];
    coeffstab = [];
    indextab = [];
    return;
end

indices = indices(:,2:4); % Drop the linear index, we don't care about it

nstats = size(values, 1);

valstab = zeros([nstats 6]);
coeffstab = zeros([nstats 6]);
indextab = zeros([nstats 6]);

% Explicitly write out all elements of the permutation group for [1 2 3]
permGroup = [1 2 3; 1 3 2; 2 1 3; 2 3 1; 3 1 2; 3 2 1];

% For every element create a list of statics data
for pg = 1:6
    P = permGroup(pg,:);

    nx = arrdims(P(1)); ny = arrdims(P(2)); nz = arrdims(P(3));
    % Standard C-style memory arrangement, plus 1-indexing to 0-indexing factor
    linind = indices(:,P(1)) + nx*indices(:,P(2)) + nx*ny*indices(:,P(3)) - nx*(ny+1);
    [indextab(:,pg) I] = sort(linind,1);
    valstab(:,pg)      = values(I);
    coeffstab(:,pg)    = coeffs(I);
end

indextab = indextab - 1; % NOTE: this is correct for GPUImogen, where CUDA indexes from 0

return;

% note: this is the old code, with every permutation written out by hand.

% Permutation [x y z]
nx = arrdims(1); ny = arrdims(2); nz = arrdims(3);
linind = indices(:,1) + nx*(indices(:,2)-1) + nx*ny*(indices(:,3)-1);
[indextab(:,1) I] = sort(linind,1);
valstab(:,1)   = values(I);
coeffstab(:,1) = coeffs(I);

% Permutation [x z y]
nx = arrdims(1); ny = arrdims(3); nz = arrdims(2);
linind = indices(:,1) + nx*(indices(:,3)-1) + nx*ny*(indices(:,2)-1);
[indextab(:,2) I] = sort(linind,1);
valstab(:,2)   = values(I);
coeffstab(:,2) = coeffs(I);

% Permutation [y x z]
nx = arrdims(2); ny = arrdims(1); nz = arrdims(3);
linind = indices(:,2) + nx*(indices(:,1)-1) + nx*ny*(indices(:,3)-1);
[indextab(:,3) I] = sort(linind,1);
valstab(:,3)   = values(I);
coeffstab(:,3) = coeffs(I);


% Permutation [y z x]
nx = arrdims(2); ny = arrdims(3); nz = arrdims(1);
linind = indices(:,2) + nx*(indices(:,3)-1) + nx*ny*(indices(:,1)-1);
[indextab(:,4) I] = sort(linind,1);
valstab(:,4)   = values(I);
coeffstab(:,4) = coeffs(I);


% Permutation [z x y]
nx = arrdims(3); ny = arrdims(1); nz = arrdims(2);
linind = indices(:,3) + nx*(indices(:,1)-1) + nx*ny*(indices(:,2)-1);
[indextab(:,5) I] = sort(linind,1);
valstab(:,5)   = values(I);
coeffstab(:,5) = coeffs(I);


% Permutation [z y x]
nx = arrdims(3); ny = arrdims(2); nz = arrdims(1);
linind = indices(:,3) + nx*(indices(:,2)-1) + nx*ny*(indices(:,1)-1);
[indextab(:,6) I] = sort(linind,1);
valstab(:,6)   = values(I);
coeffstab(:,6) = coeffs(I);

indextab = indextab - 1; % NOTE: this is correct for GPUimogen, where CUDA C indexes from 0

end
