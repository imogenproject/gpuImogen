function parImogenLoad(runFile, logFile, alias, gpuno)
% This script is the command and control manager for command line run Imogen scripts. It is
% designed to simplify the run script syntax and allow for greater extensibility.
% 
%>> runFile		run file function name to execute								str
%>> logFile		log file name for writing output information					str
%>> uid         unique identifier string for the run                            str

    %-- Initialize Imogen directory ---%
    starterRun();

    %--- Initialize MPI and GPU ---%
    context = parallel_start();
    topology = parallel_topology(context, 3);

    GIS = GlobalIndexSemantics(context, topology);

    mpi_barrier();

    % If we're running in parallel, print some bookkeeping stuff just to make
    % sure that everyone is one the same page; Share hashes to make sure that
    % exactly one processes is using each GPU.
    %
    % If serial, select the GPU indicated on the command line.
    mpiInfo = mpi_basicinfo();
    if mpiInfo(1) > 1
        if context.rank == 0; fprintf('MPI size > 1; We are running in parallel: Autoselecting GPUs\n'); end

        y = mpi_allgather(mpiInfo(2:3));
        ranks = y(1:2:end);
        hash = y(2:2:end);

        % Select ranks on this node, sort them, chose my gpu # by resultant ordering
        thisnode = ranks(hash == mpiInfo(3));
        [dump idx] = sort(thisnode);
        mygpu = idx(dump == mpiInfo(2)) - 1;
        fprintf('Rank %i/%i (on host %s) activating GPU number %i\n', context.rank, context.size, getenv('HOSTNAME'), mygpu);
        GPU_init(mygpu*2);
    else
        fprintf('MPI size = 1; We are running in serial. Activating indicated device, GPU %i\n', gpuno);
        GPU_init(gpuno);
    end

    mpi_barrier();

    runFile = strrep(runFile,'.m','');
    assignin('base','logFile',logFile);
    assignin('base','alias',alias);
    try
        eval(runFile);
    catch ME
       GPU_exit(); mpi_barrier(); mpi_finalize(); % OMG GFTO
       rethrow(ME);
    end

    GPU_exit();

    mpi_barrier();
    mpi_finalize();

    exit;
end
