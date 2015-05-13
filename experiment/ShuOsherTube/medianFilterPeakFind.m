function P = medianFilterPeakFind(Q, w, f, BC, autoplot)
% P = medianFilterPeakFind(Q, w, f, BC, autplot) searches for Q(n)  which
% are at least f times greater than mean(Q((N-w):(N+w))) and returns the
% indices of these points. BC will set boundary conditions (0 = circular,
% 1 = null) but is currently just circular. Autoplot will throw up a graph
% of Q overlaid by vertical lines marking detected peaks.

if nargin < 3; BC = 1; autoplot = 0; end
if nargin < 4; autoplot = 0; end

Qb = zeros(size(Q));

for B = -w:w
    Qb = Qb + circshift(Q, B);
end

Qb = Qb / (2*w + 1);

theta = (Q > f*Qb);

P = Q(theta);

z = zeros(size(P));

id = find(theta);

if autoplot
	plot(Q);
	hold on;
whos
	plot([id(:)'; id(:)'], [z(:)'; P(:)']);
end


end
