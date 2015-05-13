function [N V] = findLocalMaxima1D(Y, order)

if nargin == 1; order = 0; end

m = 1:numel(Y);

N=[];
V=[];

for x = 2:(numel(Y)-1)
   if (Y(x-1) < Y(x)) && (Y(x+1) < Y(x)) % There is a maximum localized within one cell of here
       if order == 0; N(end+1) = x; V(end+1) = Y(x); end
       if order == 2; [u v] = extrapMaxParabola(Y((x-1):(x+1))); N(end+1) = u+x; V(end+1)=v; end
   end
end

end

function [xm ym] = extrapMaxParabola(y)
a=y(1); b=y(2); c=y(3);

xm = .5*(a-c)/(a-2*b+c);
ym = -(a^2 + (c-4*b)^2 - 2*a*(4*b+c))/(8*(a-2*b+c));

end