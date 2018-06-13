function result = radialInvariance(sx)

if nargin == 0
    fprintf('radialInvariance is used to test the fluid component of Imogen; This routine\ntakes a saved frame from a bow shock simulation with nothing but radial outflow\nfrom the object and compares the result with 3 expected invariants.');
    result = [];
    return;
end

dims = size(sx.mass);
if numel(dims)==2
    dims(3)=1;
    alpha = 1;
else
    alpha = 2;
end

[x, y, z] = ndgrid(1:dims(1), 1:dims(2), 1:dims(3));

x = (x - ceil(dims(1)/2))*sx.dGrid{1};
y = (y - ceil(dims(2)/2))*sx.dGrid{2};
z = (z - ceil(dims(3)/2))*sx.dGrid{3};

r = sqrt(x.^2+y.^2+z.^2);

x = x ./ r;
y = y ./ r;
z = z ./ r;

g = 5/3;
l = g/(g-1);

T = .5*(sx.momX.^2+sx.momY.^2)./sx.mass;
P = (g-1)*(sx.ener - T);

result.riv1 = r.^alpha .* (sx.momX .* x + sx.momY .* y);
result.riv2 = r.^alpha .* (2*T + P);
result.riv3 = r.^alpha .* (sx.momX .* x + sx.momY .* y) .* (T + l*P) ./ sx.mass;

r0 = input('Surface radius: ');
c1 = input('Surface density: ');
c2 = input('Surface velocity: ');
c3 = input('Surface pressure: ');

result.c = r0^alpha * [c1*c2, c1*c2*c2+c3, c2*(.5*c1*c2*c2+l*c3)];

result.riv1(r < r0) = result.c(1);
result.riv2(r < r0) = result.c(2);
result.riv3(r < r0) = result.c(3);

result.R = r;

end
