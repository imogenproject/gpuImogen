function forceStop_ParallelIncompatible()

q = mpi_basicinfo();

if q(1) > 1
	error('Fatal: Calling function has explicitly marked itself parallel-incompatible; Unable to continue.');
end

end
