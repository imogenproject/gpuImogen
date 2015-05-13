function [N] = findLocalZeros1D(Y, y0, order)

if nargin == 1; y0 = 0; order = 1; end
if nargin == 2; order = 1; end

Y = Y - y0; % 
    
m = 1:numel(Y);

N=[];

for x = 2:(numel(Y)-1)
   if (Y(x-1) < 0) && (Y(x) > 0) % The target is crossed. By assumed monotonicity...
       if order == 1; N(end+1) = x+extrapZeroLin(Y((x-1):x)); end
   end
end

end

function [xm] = extrapZeroLin(y)
a=y(1); b=y(2);

xm = -a/(b-a);
end