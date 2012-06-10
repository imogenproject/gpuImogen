function kxre = kxRealAnalysis(f, dgrid, xrange)

% Snapshot of the function at the x values to test
u = f(xrange,:,:);

transdim = size(u,2);

% Shift values to best match fetures
S = zeros(numel(xrange)-1,1);

ac = zeros(transdim,1);
phi = 1:transdim;

for x = 1:numel(S)
    alpha = u(x,:);
    beta  = u(x+1,:);

    for j = 1:transdim; ac(j) = alpha * circshift(beta,[0 j])'; end

    S(x) = mean( phi(ac == max(ac)) );
end

S(S > transdim/2) = transdim - S(S > transdim/2);

kxre = S ./ diff(xrange)';


end
