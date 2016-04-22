function parImogenLoad(runFile, logFile, alias, gpuSet, nofinalize)
% This script is the command and control manager for command line run Imogen scripts. It is
% designed to simplify the run script syntax and allow for greater extensibility.
% 
%>> runFile    run file function name to execute            str
%>> logFile    log file name for writing output information str

    %-- Stand up the basics Imogen expects to be in place --%
    shutDownEverything = 0;
    failed = starterRun(gpuSet);

    if ~iscell(runFile); runFile = {runFile}; end

    if failed == 0;
            assignin('base','logFile',logFile);
            assignin('base','alias',alias);
    
            if nargin < 5;
                if mpi_myrank() == 0;
                    fprintf('No 5th argument: Assuming run is scripted, will shut down everything and mpi_finalize at completion.\n'); end
                shutDownEverything = 1;
            end

            try
                talker = mpi_amirank0();
                Nmax = numel(runFile);

                if talker
                    fprintf('==================== Received %i runfiles to execute.\n', Nmax);
                end
                for N = 1:Nmax
                    Fi = strrep(runFile{N},'.m','');
                    if mpi_amirank0(); fprintf('==================== EVALUATING %i/%i: %s.m\n',N, Nmax, Fi); end
                    eval(Fi);
                end
            
            catch ME
                fprintf('FATAL: A runfile has thrown an exception back to loader.\nRANK %i IS ABORTING JOB!\nException report follows:\n', mpi_myrank());
                prettyprintException(ME);
                if shutDownEverything;
                    fprintf('Run is non-interactive: Invoking MPI_Abort() to avoid hanging job.\n');
                    mpi_abort();
                end
            end

    else
        shutDownEverything = 1;
    end

    mpi_barrier(); % If testing n > 1 procs on one node, don't let anyone reset GPUs before we're done.

    if shutDownEverything
        clear all;         % Trashcan any remaining user GPU arrays
        GPU_ctrl('reset'); % Trashcan CUDA context
        mpi_finalize();    % Trashcan MPI runtime
        quit               % Trashcan Matlab runtime
    end


end
