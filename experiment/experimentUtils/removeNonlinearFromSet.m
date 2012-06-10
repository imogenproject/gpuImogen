function [rx ry] = removeNonlinearFromSet(x, y, reltol)
% Given a paired set of points (x_i, y_i) with nonlinearity (by some metric) N0, continue removing
% the point which most reduces N until N < reltol * N0

N0 = computeNonlinearity(x, y);

N = N0;

e = [];

iterNumber = 1;

while N > reltol*N0
    fprintf('Iteration: %i; ', iterNumber);
    for i = 1:numel(x);
        f = ones(numel(x),1); f(i) = 0; f = logical(f);
        e(i) = computeNonlinearity(x(f), y(f));
    end

    [m, ind] = min(e);
    fprintf('removing %i; ', ind)
    f = ones(numel(x),1);
    f(ind) = 0; f = logical(f);
    x = x(f);
    y = y(f);

    N = computeNonlinearity(x,y);
    fprintf('New nonlinearity: %i\n', N);

end

rx = x;
ry = y;

end

function N = computeNonlinearity(x,y)

[p S] = polyfit(x,y,1);

N = S.normr;

end
