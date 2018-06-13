ev = @(w0, kappa, ph) roots([-(1+ph), 1i*(1+ph).*kappa, w0.^2.*(1+ph), -1i*w0.^2.*kappa]);

w0 = 1;

[cc, dustload] = ndgrid(10.^(-4:.1:4), 10.^(-4:.1:1));

N = numel(cc);

omega1 = zeros(size(cc));
omega2 = zeros(size(cc));
omega3 = zeros(size(cc));

for x=1:N
    v = ev(w0, cc(x), dustload(x));
    [dump, idx] = sort(real(v));
    v=v(idx);
    omega1(x) = v(1);
    omega2(x) = v(2);
    omega3(x) = v(3);
end

