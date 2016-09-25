function forceStop_ParallelIncompatible()
% Forces Imogen to abort with an error because the calling function has marked itself
% as explicitly parallel incompatible and we are running in parallel.
q = mpi_basicinfo();

if q(1) > 1
    error('Fatal: Calling function has explicitly marked itself parallel-incompatible; Unable to continue.');
end

end
