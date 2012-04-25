function [inds vals coeffs offsets] = staticsAssemble(statVals, statCoeffs, statInds, boundaries)
% staticsAssemble takes the "manual" statics we were originally passed, and those created by the
% specification of boundary conditions, and merges them into one coherent list of indices, values
% and fade rates that can be quickly and efficiently applied.
%
% > statVals: "compiled" static value arrays, Nx6 double
% > statCoeffs: "compiled" static coefficient arrays, Nx6 double
% > statInds: "compiled" static index arrays, Nx6 double
% > boundaries: structure containing statics (if any) created by the boundary conditions
%               3 cells for X, Y and Z specific BCs, which are only applied when fluxing in
%               that direction
% < inds: The compiled list of indices, Nx6
% < vals: The compiled list of values, Nx6
% < coeffs: The compiled list of coefficients, Nx6
% < offsets: 12x1 list containing six pairs of [offset size] to index the arrays of statics

bdyIdx = 0;

% The output arrays
vals = []; coeffs = []; inds = [];
offsets = [];

% For each of 6 possible ways to uniquely permute [XYZ], which are
% [xyz], [xzy], [yxz], [yzx], [zxy], [zyx]

for i = 1:6
    % First compute the offset to get to here in the array, ie the number of elements already in it
    offsets(2*i-1) = size(vals,1);

    % Append the user defined statics, if any
    if(numel(statInds) > 0)
        vals   = [vals; statVals(:,i)];
        coeffs = [coeffs; statCoeffs(:,i)];
        inds   = [inds; statInds(:,i)];
    end

    % Make sure we stick the i-direction boundary condition statics with the i-first indices,
    % i.e. only append x-specific BCs if the first (being-fluxed) index is X. Note above that the
    % sequences start with XXYYZZ
%    if mod(i,2) == 1; bdyIdx = bdyIdx + 1; end


    % Append statics created by our boundary conditions
    if(numel(boundaries.index) > 0)
 %       vals   = [vals; boundaries(bdyIdx).value(:,i)];
        vals   = [vals; boundaries.value(:,i)];
        coeffs = [coeffs; boundaries.coeff(:,i)];
        inds   = [inds; boundaries.index(:,i)];
    end

    % Store the number of statics for this flux direction
    offsets(2*i) = size(vals,1) - offsets(2*i-1); 
end

end 
