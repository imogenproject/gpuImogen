function [iout, Pout] = medianFilterPeakFind(Q, w, f, log, BC, autoplot)
% P = medianFilterPeakFind(Q, w, f, log, BC, autplot) searches for Q(n)  which
% are at least f times greater than mean(Q((N-w):(N+w))) and returns the
% indices of these points. BC will set boundary conditions (0 = circular,
% 1 = null) but is currently just circular. Autoplot will throw up a graph
% of Q overlaid by vertical lines marking detected peaks.
% if 'log' is not zero, looks for Q - mean(...) > f instead.

if nargin < 3; log = 0; end
if nargin < 4; BC = 1; autoplot = 0; end
if nargin < 5; autoplot = 0; end

Q = Q(:);

Qb = zeros(size(Q));

% Compute medians
for B = -w:w
    Qb = Qb + circshift(Q, B);
end
Qb = Qb / (2*w + 1);

if BC % if not circular: chop edge values
    m = zeros([w+1, 2*w+1]);

    for N = 1:(w+1)
       m(N,1:(w+N))=1/(w+N+1);
    end
    
    Qb(1:(w+1)) = m*Q(1:(2*w+1));
    
    m = zeros([w+1, 2*w+1]);

    for N = 1:(w+1)
       m(N,(end-w+N):end)=1/(w+N+1);

    end
    
    Qb((end-w):end) = m*Q((end-2*w):end);
end

if log
    theta = (Q - Qb > f);
else
    theta = (Q > f*Qb);
end

P = Q(theta);

id = find(theta);

iout(1) = id(1);
Pout(1) = P(1);
% Loop over id values, binning when we have adjacent values
M = 1;
for N = 2:numel(id)
    
    if id(N) ~= id(N-1)+1; % if not adjacent,
        M=M+1;
    end
    
    iout(M) = id(N);
    Pout(M) = P(N);
end

if autoplot
	plot(Q);
	hold on;
    z = zeros(size(Pout));
	plot([iout(:)'; iout(:)'], [z(:)'; Pout(:)']);
end


end
