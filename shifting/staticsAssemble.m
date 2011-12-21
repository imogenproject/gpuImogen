function [vals coeffs inds offsets] = staticsAssemble(statVals, statCoeffs, statInds, boundaries)
% stat variables: the statics assigned to the array
% boundaries: the statics (if any) assigned to the array by the boundary conditions

bdyIdx = 0;

vals = []; coeffs = []; inds = [];
offsets = [];

for i = 1:6
    % First the offset to get to here in the array
    offsets(2*i-1) = size(vals,1);

    % Append the "real" statics
    if(numel(statInds) > 0)
        vals   = [vals; statVals(:,i)];
        coeffs = [coeffs; statCoeffs(:,i)];
        inds   = [inds; statInds(:,i)];
    end

    % Make sure we stick the i-direction boundary condition statics with the i-first indices
    if mod(i,2) == 1; bdyIdx = bdyIdx + 1; end

    % Append boundary condition statics
    if(numel(boundaries(bdyIdx).index) > 0)
        vals   = [vals; boundaries(bdyIdx).value(:,i)];
        coeffs = [coeffs; boundaries(bdyIdx).coeff(:,i)];
        inds   = [inds; boundaries(bdyIdx).index(:,i)];
    end

    offsets(2*i) = size(vals,1) - offsets(2*i-1); 
end

end 
