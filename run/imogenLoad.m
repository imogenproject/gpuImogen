function imogenLoad(runFile, logFile, alias, gpuno)
% This script is the command and control manager for command line run Imogen scripts. It is
% designed to simplify the run script syntax and allow for greater extensibility.
% 
%>> runFile		run file function name to execute								str
%>> logFile		log file name for writing output information					str
%>> uid         unique identifier string for the run                            str

    %-- Initialize Imogen directory ---%
    starterRun();
 
    context = parallel_start();
    topology = parallel_topology(context, 3);

    GIS = GlobalIndexSemantics(context, topology);

    runFile = strrep(runFile,'.m','');
    assignin('base','logFile',logFile);
    assignin('base','alias',alias);

    gm = GPUManager.getInstance();
    
    gm.init(gpuno, 3, 1);

    try
        eval(runFile);
    catch ME
       rethrow(ME);
    end

    mpi_barrier();
    mpi_finalize();

    exit();

end
