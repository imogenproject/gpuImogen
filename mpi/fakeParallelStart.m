function [context topo] = fakeParallelStart()
% If it is for some reason called for, fake the output values from real
% calls to parallel_start() and parallel_topology() as for an mpi-serial
% program.
context.rank = 0;
context.size = 1;
context.comm = 0;
context.request = 0;
context.error = 0;

topo.ndim = 3;
topo.comm = 0;
topo.coord = [0 0 0];
topo.neighbor_left = [0 0 0];
topo.neighbor_right = [0 0 0];
topo.nproc = [1 1 1];


end
