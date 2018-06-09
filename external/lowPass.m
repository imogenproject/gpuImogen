function lp = lowPass(f, c, Nini)

if nargin < 3; Nini = 1; end

lp = zeros(size(f));
lp(1) = mean(f(1:Nini));

for t = 2:numel(f)
    lp(t) = lp(t-1) + c*(f(t) - lp(t-1));
end

lp = reshape(lp, size(f));

end