function y = lowpassFilter(x, k0)

y = zeros(size(x));
y(1) = x(1);

for u = 2:numel(x)
    y(u) = y(u-1) + k0*(x(u)-y(u-1));
end

end
