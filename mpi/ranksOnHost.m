function N = ranksOnHost()
% Identifies how many MPI ranks are running on this host
% and which one of them I am. Returns
% [# of hosts; which I am]

x = mpi_basicinfo();

pertinent = x(2:3); % rank and hostname hash

everyone = mpi_allgather(pertinent);

everyone = reshape(everyone, [2 x(1)]);

allYall = find(everyone(2,:) == x(3));

N(1) = numel(allYall);

% Find out how many ranks are before me
myrank = x(2);

them = numel(find(everyone(2,1:(myrank)) == x(3)));

N(2) = them;


end
