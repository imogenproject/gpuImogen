function frontX = trackFront2(mass, x, ycrit)
% x = trackFront2(rho, x, ycrit)
% Given a saveframe from Imogen, Identifies the shock's position.

% Take dRho/dx and find the indices of the max
d = diff(mass, 1, 1);

yy = circshift(mass(1:(end-1),:,:),10);

d = d .* (abs(yy-1) < .1);

[dRho_dx, ind] = max(d, [], 1);
%dRho_dx = dRho_dx .* diag(mass(ind,:,:))';

if nargin < 3
    halfval = (mass(end,1,1) + mass(1,1,1))/2;
else
    halfval = ycrit;
end

% Get rho in the cell before the max derivative
% Linearly extrapolate to where it's between equilibrium values
f0 = mass(ind + size(mass,1)*(0:(size(mass,2)-1)) );
dx = (halfval - f0) ./ dRho_dx;

% Convert the linear indices to X indices and add the delta we found before
a = x(mod(ind, size(mass, 1)));
b = x(mod(ind, size(mass, 1))+1);

frontX = a + (b-a).*dx;

end
