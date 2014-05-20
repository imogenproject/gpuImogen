function result = fim(x, func)
% Used a corrected Fourier method to calculate the integral of func(x) at all points x
% with F(1) = 0

f = func(x);

h = x(2)-x(1);
w=numel(x)*h;

x = x - x(1); % Make a convenient transform to simplify

%c0 = (f(end)+h*(f(end)-f(end-1)) - f(1))/w
c0 = 0;%(f(end)-f(1))/w
g = c0*x;

f = f - g;

N = numel(f);
ftilde = fft(f);

if floor(N/2) == N/2
    coefs = 1i*[[1 1:N/2] -[(N/2-1):-1:1]];
else
    coefs = 1i*[[1 1:(floor(N/2))] -[(floor(N/2)):-1:1]];
end

% Peel off 0th Fourier mode which is not integrable this way
fbar = ftilde(1)/N;
ftilde(1) = 0;

F = real(w*ifft(ftilde./coefs)/(2*pi));

result = x.*(fbar + x*c0/2) + F;

% Return the promised value
result = result - result(1);


end
