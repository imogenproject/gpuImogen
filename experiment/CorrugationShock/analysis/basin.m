
x = 0:.01:1;
y = exp((-2.5 - 5*1i)*x) + .4 * exp((-1.8 - 10*1i)*x);

%[kre kim] = ndgrid(-8:.08:-.03, -25:.2:25);

e = zeros(size(kre));
f = zeros(size(kre));
f2 = zeros(size(f));
g = zeros(size(f));
g2= zeros(size(f));
h = zeros(size(g));

%parfor n = 1:numel(f);
%    [f(n) g(n) h(n)]
    [u v h]  = multiwaveFit(y, .01, [kre + 1i*kim, -1.8 - 10*1i], [1, .4]);
%    f(n) = u(1); f2(n) = u(2);
%    g(n) = v(1); g2(n) = v(2);

%     [u h(n)] = multiwave_gradient(y, .01, [kre(n) + 1i*kim(n), -1.6 - 7*1i], [1, .5]);
%     f(n) = u(1);
%     g(n) = u(2);
%     e(n) = u(5);
%end

return;

f(abs(f) > 50) = 1;
f2(abs(f2) > 50) = 1;
h(isnan(h)) = 1;
h(abs(h) > 1e5) = 0;

figure(); imagesc(log(h));

figure();
subplot(1,2,1);
  surf(kre,kim,real(f),log(h));
  xlabel('kre');
  ylabel('kim'); title('real, k1');
subplot(1,2,2);
  surf(kre,kim,imag(f),log(h));
  xlabel('kre');
  ylabel('kim'); title('imaginary, k1');

figure();
subplot(1,2,1);
  surf(kre,kim,real(f2),log(h));
  xlabel('kre');
  ylabel('kim'); title('Real, k2');
subplot(1,2,2);
  surf(kre,kim,imag(f2),log(h));
  xlabel('kre');ylabel('kim'); title('imaginary, k2');


