function mpi_fakestart(dims, grid)
% FIXME: Is this even required anymore now hat I've replaced these fuctions?
c = parallel_start();
t = parallel_topology(c, dims);

GIS = GlobalIndexSemantics(c, t);
GIS.setup(grid);

end

