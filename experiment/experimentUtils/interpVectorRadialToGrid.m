function [Rx, Ry, Rz] = interpVectorRadialOntoGrid(R, F_r, rrange, x, y, z, F0x, F0y, F0z)

if nargin == 9; Rx = 1.0*F0x; Ry = 1.0*F0y; Rz = 1.0*F0z; else; Rx = zeros(size(x)); Ry = Rx; Rz=Rx; end

% This is probably a re-calculation, c'est la vie
norm = sqrt(x.^2+y.^2+z.^2);

% Parts inside the smallest R if they exist
if rrange(1) < R(1)
    s = (norm >= rrange(1)) & (norm < R(1));
    Rx(s) = F_r(1) * x(s) ./ norm(s);
    Ry(s) = F_r(1) * y(s) ./ norm(s);
    Rz(s) = F_r(1) * z(s) ./ norm(s);
end

% Parts between the smallest R and either the largest or the max to interpolate
s = (norm > R(1)) & (norm <= min(rrange(2), R(end)) );
Rx(s) = interp1(R, F_r, norm(s)) .* x(s) ./ norm(s);
Ry(s) = interp1(R, F_r, norm(s)) .* y(s) ./ norm(s);
Rz(s) = interp1(R, F_r, norm(s)) .* z(s) ./ norm(s);

% Parts beyond the outer edge of the calculated radial function
if rrange(2) > R(end)
    s = (norm > R(end)) .* (norm < rrange(2));
    R_x(s) = F_r(end) * x(s) ./ norm(s);
    R_y(s) = F_r(end) * y(s) ./ norm(s);
    R_z(s) = F_r(end) * z(s) ./ norm(s);
end


end
