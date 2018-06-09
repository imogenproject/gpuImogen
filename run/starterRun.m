function failed = starterRun(gpuSet)
% This routine does the initialization to include the imogen program paths and stand up the MPI
% and GPU routines necessary for new runs.

    % Attempt locate path from location of this m-file and make sure Imogen paths
    % are in the loop
    try
        rootPath = [strrep(mfilename('fullpath'),'starterRun','') '../'];
        cd(rootPath);
    catch MERR
    end
        
    found = false;
    for i=1:3
        
        files = dir(cd);
        for n=1:length(files)
            if strcmp(files(n).name,'imogen.m'), found = true; break; end
        end
        
        if found, break;
        else; cd('..');
        end
    end

    try
        includeImogenPaths();    %Initialize directory structure
    catch MERR
        error('Imogen:starterRun:ImogenRootFailure','Unable to find Imogen root directory. Run aborted.');
    end


    % Get us ready to talk to MPI
    if ~mpi_isinitialized()
        % Start up PGW\
        % defaults to a 3D topology
        context = parallel_start();
        topology = parallel_topology(context, 3);

        % This will be init'd & stores itself as a persistent var
        parInfo = ParallelGlobals(context, topology); %#ok<NASGU>
        
        if context.rank==0
            fprintf('\n---------- MPI Startup\nFirst start: MPI is now ready. Roll call:\n'); end
        mpi_barrier();
        fprintf('Rank %i ready.\n', int32(context.rank));
        mpi_barrier();
    else
        if mpi_amirank0()==0; fprintf('---------- MPI Startup\nMPI is already ready.'); end
    end

% If debugging compiled code:
%debugSpin(); % will make all ranks spin
%debugspin([list of rank #s]); // will make only those ranks spin

    % When the GPU kernels are compiled, the Makefile drops integer zero into this
    % file if SSPRK is not used and integer one if it is.
    % SSPRK requires 4 halo cells, explicit midpoint requires 3.
    % Safely default to 4 if somehow it doesn't exist.
    try haloSize = csvread('.fluidMethod'); catch ohwell; haloSize = 4; end

    %--- Acquire GPU manager class, set GPUs, and enable intra-node UVM
    gm = GPUManager.getInstance();

    dimensionDistribute = 2;
    teslaCards = selectGPUs(gpuSet);

    gm.init(teslaCards, haloSize, dimensionDistribute);

    mpi_barrier();
    if mpi_amirank0(); disp('Testing device usability in indicated configuration...'); end

    testData = rand([32 32 32]);
    worked = 0;
    try
        testGPU = GPU_Type(testData);
    catch awcrap
        prettyprintException(awcrap, 0, 'Exception generated attempting to simply upload data to GPUs\nAborting job.\n');
        worked = mpi_myrank() + 1;
    end
    
    worked = mpi_max(worked);
    if worked > 0
        failed = 1;
    else
        if ~gm.isInitd
            GPU_ctrl('peers',1);
        end
        failed = 0;
    end

    return;
end
