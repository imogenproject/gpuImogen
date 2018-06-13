function frontX = trackFrontHydro(mass, x, y0)
% Given a saveframe from Imogen, Identifies the shock's position.

% Take dRho/dx and find the indices of the max
jmp = (mass > y0);
d = diff(jmp, 1, 1);
[~, ind] = max(d, [], 1);

% Stupid edge case BS
ind(ind == numel(mass)) = ind(ind == numel(mass)) - 1;

halfval = y0;
% Get rho in the cell before the max derivative
% Linearly extrapolate to where it's between equilibrium values
f0 = mass(ind);
dx = (y0 - f0) ./ (mass(ind+1)-mass(ind));

% Convert the linear indices to X indices and add the delta we found before
a = x(mod(ind, size(mass, 1)))';
b = x(mod(ind, size(mass, 1))+1)';

frontX = a + (b-a).*dx;

end
