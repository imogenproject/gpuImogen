function y = squish(x, behavior)
% Replacement for squeeze that behaves consistently with 1xN
% (removing the leading 1)
d = size(x);
%c = cumprod(size(x));

if nargin == 1
    % Default to 'rational' behavior of removing all singleton dimensions
    d = d(d>1);
else
    if strcmp(behavior,'onlyleading')
    % This is mainly to deal with an idiotic legacy behavior of storing
    % 3D vector arrays as as size [3 nx ny nz] array
    % Just.. ugh.
        uno = (d > 1);
        d = d(uno | (cumprod(d) > 1));
    end
    if strcmp(behavior,'all')
        d = d(d>1);
    end
end

if isempty(d); d = 1; end
if numel(d) == 1; d(2) = 1; end

y = reshape(x, d);

end
