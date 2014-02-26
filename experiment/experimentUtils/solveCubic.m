function x = solveCubic(a, b, c, d)

p = (3*a*c-b*b)/(3*a^2);
q = (2*b^3-9*a*b*c+27*a^2*d)/(27*a^3);

% Now have t^3 + p t + q = 0

w3 = (-q + sqrt(q^2 + 4*p^3/27))/2;

modw = abs(w3)^(1/3);

w = [1 (1+sqrt(-3))/2 (1-sqrt(-3))/2] * modw;

t = w - p ./ (3*w);

x = t - b/(3*a)

end
