function frontX = trackFront2(qty, x, ycrit)
% x = trackFront2(rho, x, ycrit)
% Given a saveframe from Imogen, Identifies the shock's position.

% Take dRho/dx and find the indices of the max
d = diff(qty, 1, 1);

yy = circshift(qty(1:(end-1),:,:),3);

d = d .* (abs(yy-1) < .1);

[dRho_dx, ind] = max(d, [], 1);
%dRho_dx = dRho_dx .* diag(mass(ind,:,:))';

if nargin < 3
    halfval = (qty(end,1,1) + qty(1,1,1))/2;
else
    halfval = ycrit;
end

% Get value in the cell before the max derivative
% Linearly extrapolate to where it's between equilibrium values
f0 = qty(ind + size(qty,1)*(0:(size(qty,2)-1)) );
dx = (halfval - f0) ./ dRho_dx;

% These are clearly wrong, prevent nonsense
dx(dx > 1) = 0; 
dx(dx < 0) = 0;

% Convert the linear indices to X indices and add the delta we found before
a = x(mod(ind, size(qty, 1)));
b = x(mod(ind, size(qty, 1))+1);

frontX = a + (b-a).*dx;

end
