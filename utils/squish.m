function y = squish(x)
% Replacement for squeeze that non-retardedly handles (1xN) -> (Nx1)

d = size(x);
d = d(d>1);
if isempty(d); d = 1; end
if numel(d) == 1; d(2) = 1; end

y = reshape(x, d);

end
