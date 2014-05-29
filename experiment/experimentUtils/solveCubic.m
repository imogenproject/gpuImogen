function x = solveCubic(a, b, c, d)

p = -b/(3*a);
q = p^3 + (b*c-3*a*d)/(6*a^2);
r = c/(3*a);

L = q + sqrt(q^2 + (r-p^2)^3);
M = q - sqrt(q^2 + (r-p^2)^3);

al = angle(L)/3;
am = angle(M)/3;

x = abs(L)^(1/3) * exp(1i*(al+[0 2 4]*pi/3)) + abs(M)^(1/3) * exp(1i*(am+[0 2 4]*pi/3)) + p;

% For great justice, finish off with one round of newton-raphson
x = x - (d + x.*(c + x.*(b + a*x)))./(c + x.*(2*b + 3*a*x));
end
