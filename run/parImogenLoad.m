function parImogenLoad(runFile, logFile, alias)
% This script is the command and control manager for command line run Imogen scripts. It is
% designed to simplify the run script syntax and allow for greater extensibility.
% 
%>> runFile		run file function name to execute								str
%>> logFile		log file name for writing output information					str
%>> uid         unique identifier string for the run                            str

    %-- Initialize Imogen directory ---%
    starterRun();

    %--- Initialize MPI and GPU ---%
    mpi_init();
    basics = mpi_basicinfo();
    if basics(1) > 1 % If this is in fact parallel, autoinitialize GPUs
                     % Otherwise initialize.m will choose one manually
        fprintf('MPI size > 1; We are running in parallel; Autoinitializing GPUs\n');

        y = mpi_allgather(basics(2:3));
        
        ranks = y(1:2:end);
        hash = y(2:2:end);

        % Select ranks on this node, sort them, chose gpu # by resultant ordering
        thisnode = ranks(hash == basics(3));
        [dump idx] = sort(thisnode);
        mygpu = idx(dump == basics(2));
        fprintf('Rank %i on activating GPU number %i\n', basics(2), mygpu-1);
        GPU_init(mygpu-1);
    end

    runFile = strrep(runFile,'.m','');
    assignin('base','logFile',logFile);
    assignin('base','alias',alias);
    try
        eval(runFile);
    catch ME
       rethrow(ME);
    end

    enderRun();

    mpi_finalize();

    exit;
end
