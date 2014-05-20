function result = fim(x, func)
% Used a corrected Fourier method to calculate the integral of func(x) at all points x
% with F(1) = 0

f = func(x);

h = x(2)-x(1);
w=numel(x)*h;

xend = x(end)+h;
xst  = x(1);

x = x - x(1); % Make a convenient transform to simplify the polies

% Evaluate f, f', f'' at the endpoints
fE  = func(xend);
 fpS = imag(func(xst+1i*eps))/eps;
 fpE = imag(func(xend+1i*eps))/eps;
  hvec = sqrt(1i)*[-1 -.5 .5 1]/32768;
  fcmp = func(xst + hvec);
  fppS = 1073741824*imag(64*(fcmp(3)+fcmp(2))-fcmp(1)-fcmp(4))/15;
  fcmp = func(xend+ hvec);
  fppE = 1073741824*imag(64*(fcmp(3)+fcmp(2))-fcmp(1)-fcmp(4))/15;

% Order of continuity we wish to have
switch 3;
  case 0; c1 = 0; c2 = 0; c3 = 0;
  case 1; c1 = (fE-f(1))/w;
          c2 = 0; c3 = 0;
  case 2; c1 = -(2*f(1) - 2*fE - w*fpS + w*fpE)./(2*w);
          c2 = (fpE-fpS)./(2*w);
          c3 = 0;
  case 3; c1 = (12*fE - 6*fpE*w + 6*fpS*w + fppE*w.^2 - fppS*w.^2 - 12*f(1))./(12*w);
          c2 = (2*fpE - 2*fpS - fppE*w + fppS*w)./(4*w);
          c3 = (fppE - fppS)./(6*w);
end

g = x.*(c1 + x.*(c2 + c3*x));

f = f - g;

N = numel(f);
ftilde = fft(f);

if floor(N/2) == N/2
    coefs = 1i*[[1 1:N/2] -[(N/2-1):-1:1]];
else
    coefs = 1i*[[1 1:(floor(N/2))] -[(floor(N/2)):-1:1]];
end

% Peel off 0th Fourier mode which is not integrable this way
c0 = ftilde(1)/N;
ftilde(1) = 0;

F = real(w*ifft(ftilde./coefs)/(2*pi));

result = x.*(c0 + x.*(c1/2 + x.*(c2/3+x*c3/4))) + F;

% Return the promised value
result = result - result(1);


end
