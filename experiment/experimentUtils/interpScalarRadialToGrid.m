function result = interpScalarRadialOntoGrid(R, F_r, rrange, x, y, z, f0)

if nargin == 7; result = 1.0*f0; else; result = zeros(size(x)); end

% This is probably a re-calculation, c'est la vie
norm = sqrt(x.^2+y.^2+z.^2);

% Parts inside the smallest R if they exist
if rrange(1) < R(1)
    s = (norm >= rrange(1)) & (norm < R(1));
    result(s) = F_r(1);
end

% Parts between the smallest R and either the largest or the max to interpolate
s = (norm > R(1)) & (norm <= min(rrange(2), R(end)) );
result(s) = interp1(R, F_r, norm(s));

% Parts beyond the outer edge of the calculated radial function
if rrange(2) > R(end)
    s = (norm > R(end)) .* (norm < rrange(2));
    result(s) = F_r(end);
end


end
