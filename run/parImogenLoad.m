function parImogenLoad(runFile, logFile, alias, gpuSet, nofinalize)
% This script is the command and control manager for command line run Imogen scripts. It is
% designed to simplify the run script syntax and allow for greater extensibility.
% 
%>> runFile    run file function name to execute            str
%>> logFile    log file name for writing output information str

    %-- Stand up the basics Imogen expects to be in place --%
    shutDownEverything = 0;
    failed = starterRun(gpuSet);

    if failed == 0;
        % Accept magic name 'gputest' to simply hammer the GPUs with work rather than run Imogen
        if strcmp(runFile,'gputest')
            shutDownEverything = 1;
            cd gpuclass; % tortureTest expects to be run from here
            try
                tortureTest(gpuSet, 'y', 4);
            catch crap
                prettyprintException(crap, 1, 'One or more ranks have failed GPU tests! ECC -> hardware problem. Other -> Bad code checked into git?');
                failed = 1;
            end
            
            failed = mpi_max(failed);
            if(failed) mpi_abort(); end
        else

            runFile = strrep(runFile,'.m','');
            assignin('base','logFile',logFile);
            assignin('base','alias',alias);
    
            if nargin < 5;
                if mpi_myrank() == 0;
                    fprintf('No 5th argument: Assuming run is scripted, will shut down everything and mpi_finalize at completion.\n'); end
                shutDownEverything = 1;
            end

            try
                eval(runFile);
            catch ME
                fprintf('FATAL: Runfile has thrown an exception back to loader.\nRANK %i IS ABORTING JOB!\nException report follows:\n', mpi_myrank());
                prettyprintException(ME);
                if shutDownEverything;
                    fprintf('Run is non-interactive: Invoking MPI_Abort() to avoid hanging job.\n');
                    mpi_abort();
                end
            end

        end
    else
        shutDownEverything = 1;
    end

    mpi_barrier(); % If testing n > 1 procs on one node, don't let anyone reset GPUs before we're done.
    if shutDownEverything
        GPU_ctrl('reset');
        mpi_finalize();
        clear all;
    end


end
