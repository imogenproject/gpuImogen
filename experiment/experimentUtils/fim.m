function result = fim(x, y)
% > x: Set of points, uniformly spaced
% > y: @(x) function(x) to integrate
% Use endpoint-corrected Fourier method to calculate
% the integral F(x) with F(1)=0

f = y(x);

h = x(2)-x(1);
w=numel(x)*h;

xend = x(end)+h;
xst  = x(1);

x = x - x(1); % Make a convenient transform to simplify the polies

% Evaluate f, f' @ ends to machine precision, f'' to ~1e-10
% http://www.sciencedirect.com/science/article/pii/S0377042707004086
fE  = y(xend);
 fpS = imag(y(xst+1i*eps))/eps;
 fpE = imag(y(xend+1i*eps))/eps;
  hvec = sqrt(1i)*[-1 -.5 .5 1]/32768;
  fcmp = y(xst + hvec);
  fppS = 1073741824*imag(64*(fcmp(3)+fcmp(2))-fcmp(1)-fcmp(4))/15;
  fcmp = y(xend+ hvec);
  fppE = 1073741824*imag(64*(fcmp(3)+fcmp(2))-fcmp(1)-fcmp(4))/15;

% Order of continuity we wish to have
switch 3
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

% Create aux polynomial and subtract
g = x.*(c1 + x.*(c2 + c3*x));
f = f - g;

N = numel(f);
ftilde = fft(f);

% Write out spectral integral transform in kspace
if floor(N/2) == N/2
    coefs = 1i*[[1 1:N/2] -[(N/2-1):-1:1]];
else
    coefs = 1i*[[1 1:(floor(N/2))] -[(floor(N/2)):-1:1]];
end

% Peel off 0th Fourier mode 
c0 = ftilde(1)/N;
ftilde(1) = 0;

% Transform integral back to realspace and add int. of aux poly
F = real(w*ifft(ftilde./coefs)/(2*pi));
result = x.*(c0 + x.*(c1/2 + x.*(c2/3+x*c3/4))) + F;

result = result - result(1);

end
