function context = parallel_start()

hot = mpi_init();

if hot
   warning('Note: MPI has already been initialized. Calling this is probably wrong.'); 
end

v = mpi_basicinfo();

context = struct('rank', v(2), 'size', v(1), 'comm', 0, 'request', 0, 'error', 0);

end