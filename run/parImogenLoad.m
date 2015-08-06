function parImogenLoad(runFile, logFile, alias, gpuSet, nofinalize)
% This script is the command and control manager for command line run Imogen scripts. It is
% designed to simplify the run script syntax and allow for greater extensibility.
% 
%>> runFile    run file function name to execute            str
%>> logFile    log file name for writing output information str

% DEVELOPERS: Uncomment to buy time to attach GDB to parallel runs
% disp('FIVE SECOND WARNING');
% pause(5);



    %-- Stand up the basics Imogen expects to be in place --%
    starterRun(gpuSet);

    runFile = strrep(runFile,'.m','');
    assignin('base','logFile',logFile);
    assignin('base','alias',alias);
    
    shutDownEverything = 0;
    if nargin < 5;
        if mpi_myrank() == 0; fprintf('No 5th argument: Assuming run is scripted, will shut down everything and mpi_finalize at completion.\n'); end
        shutDownEverything = 1;
    end

    try
        eval(runFile);
    catch ME
        ME
        ME.cause
        for n = 1:numel(ME.stack);
            fprintf('In %s:%s at %i\n', ME.stack(n).file, ME.stack(n).name, ME.stack(n).line);
        end
        fprintf('DISASTER: Evaluation of runfile failed for me; rank %i aborting!\n', GIS.context.rank);
        shutDownEverything = 1;
    end

    mpi_barrier();
    if shutDownEverything
        clear GPUManager GlobalIndexSemantics
        GPU_ctrl('reset');
        mpi_finalize();
    end
end
