function mpi_fakestart(dims, grid)

c = parallel_start();
t = parallel_topology(c, dims);

GIS = GlobalIndexSemantics(c, t);
GIS.setup(grid);

end

