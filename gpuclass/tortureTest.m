function tortureTest(multidev, dorad, nTests)
% tortureTest(devicelist, 'y'/'n' to radiative cooling test, numTests)
% > device list: Set of integers naming GPUs to use, as enumerated by GPU_ctrl('info')
% > dorad: If == 'y' tests cudaFreeRadiation
% > numTests: How many times to fuzz each function

if nargin < 1
    multidev = 0;
    disp('>>> WARNING: No input array of device IDs given; Defaulting to [0] and disabling multi-device');
end

% if we are running in a SLURM env
if multidev == -1; multidev = selectGPUs(-1); end

if numel(unique(multidev)) ~= numel(multidev)
    disp('>>> FATAL: The same device is named repeatedly.');
    error('Multiple devices: Device IDs must be unique.');
end

if nargin < 2
    dorad = 'y';
    disp('>>> WARNING: Radiation test defaulting to WILL HAPPEN');
end

if nargin < 3
    nTests = 5;
    disp('>>> WARNING: Number of tests not indicated, defaulting to 5 ea');
end

addpath('../mpi');
mpi_init();
if mpi_amirank0()
    disp('Started MPI');
    disp('#########################');
    disp('Setting up test environment');
    disp('#########################');
end

GPU_ctrl('peers',1);
if mpi_amirank0(); disp('Turned on peer memory access'); end

isparallel = mpi_basicinfo(); isparallel = (isparallel(1) > 1);

if numel(multidev) > 1
    disp('	>>> Using the following devices for multidevice testing <<<');
    disp(multidev);
end

if mpi_amirank0(); disp('#########################');
                   disp('Running basic funtionality tests');
                   disp('#########################');
end


% Check that the basic stuff works first
x = GPUManager.getInstance();

basicunits = basicTest(multidev);
if basicunits > 0
    if isparallel
        disp('Rank %i got fatal problems in basic tests: Rank will not run further tests.\n');
    else
        error('Fatal problem: Unit tests of basic functionality failed. Aborting further tests.');
    end
end

if mpi_amirank0(); disp('#########################'); end

singledev = multidev(1);

% Kernels used by Imogen:
% TIME_MANAGER	cudaSoundspeed directionalMaxFinder
% 100% coverage by unit tests
% FLUX
% 	ARRAY_INDEX_EXCHANGE:	cudaArrayRotateB
%	RELAXING_FLUID:		freezeAndPtot, cudaFluidStep, cudaHaloExchange
% 75% coverage by unit tests
%	-> cudaFluidStep can be tested by run_FullTestSuite
% SOURCE
%	cudaSourceRotatingFrame, cudaAccretingStar, cudaSourceScalarPotential, cudaFreeRadiation
% 50% coverage by unit tests (accreting star broken)

if numel(multidev) < 2
    functests = [1 0 0 0];
    if mpi_amirank0(); disp('>>> WARNING: Only one device indicated for use. Multi-device fuzzing tests will not be done'); end
else
    functests = [1 1 1 1];
end

% parallel compat: jump through the mpi_barrier()s but do nothing
if basicunits > 0; functests = [0 0 0 0]; end

names = {'cudaArrayAtomic', 'cudaArrayRotateB', 'cudaSoundspeed', 'directionalMaxFinder', 'cudaSourceScalarPotential', 'cudaSourceRotatingFrame'};

if dorad == 'y'; names{end+1} = 'cudaFreeRadiation'; end

if mpi_amirank0()
    for N = 1:numel(names); disp(['	' names{N}]); end
    disp('NOTE: setup & data upload times dwarf execution times here; No speedup will be observed.');
    disp('#########################');
end

randSeed = 5418;
% Note these do not DEFINE convenient roundish sizes, actual sizes are randomly chosen up to these
res2d = [2048 2048 1];
res3d = [192 192 192];

% Test single-device operation
printit = (isparallel*[1 1 1 1] | functests) * mpi_amirank0();

pm = ParallelGlobals();

if printit(1); disp('==================== Testing on one GPU'); end
if functests(1)
    x.init(singledev, 3, 1); rng(randSeed);
    onedev = unitTest(nTests, res2d, nTests, res3d, names);
else; onedev = 0;
end

mpi_barrier();

% Test two partitions on different devices
if printit(2); disp('==================== Testing multiple GPUs, X partitioning'); end
if functests(2)
    x.init(multidev, 3, 1); rng(randSeed);
    if (numel(x.deviceList) > 1) && (pm.topology.nproc(x.partitionDir) == 1); extHalo = 1; else; extHalo = 0; end
    x.useExteriorHalo = extHalo;
    devx = unitTest(nTests, res2d, nTests, res3d, names);
else; devx = 0;
end

mpi_barrier();

if printit(3); disp('==================== Testing multiple GPUs, Y partitioning'); end
if functests(3)
    x.init(multidev, 3, 2); rng(randSeed);
    if (numel(x.deviceList) > 1) && (pm.topology.nproc(x.partitionDir) == 1); extHalo = 1; else; extHalo = 0; end
    x.useExteriorHalo = extHalo;
    devy = unitTest(nTests, res2d, nTests, res3d, names);
else; devy = 0;
end

mpi_barrier();

if printit(4); disp('==================== Testing multiple GPUs, Z partitioning'); end
if functests(4)
    x.init(multidev, 3, 3); rng(randSeed);
    if (numel(x.deviceList) > 1) && (pm.topology.nproc(x.partitionDir) == 1); extHalo = 1; else; extHalo = 0; end
    x.useExteriorHalo = extHalo;
    devz = unitTest(0, res2d, nTests, res3d, names);
else; devz = 0;
end

mpi_barrier();
if mpi_amirank0(); disp('#########################'); disp('RESULTS:'); end

tests = [onedev devx devy devz];

if isparallel
    ambad  = mpi_max(max(tests)); % did ANYONE fail?
    amgood = mpi_min(min(tests)); % did anyone succeed?

    if ambad == 0
        if mpi_amirank0(); disp('  >>> PARALLEL RESULT SUMMARY: ALL NODES PASSED <<<'); end
    end

    if (ambad > 0) && (amgood == 0) % Different good/bad results on different nodes!!!
        if mpi_amirank0(); disp('  >>> PARALLEL PROBLEMS: SOME NODES PASSED AND OTHERS FAILED <<<'); end
    end

    if amgood > 0 % everyone failed:
        tests = mpi_max(tests);
        if mpi_amirank0(); fprintf('  >>> PARALLEL RESULT: ALL RANKS FAILED. RECOMMEND RERUN IN SERIAL <<<\n  >>> DISPLAYING UNIT TEST OUTPUT FOR max(tests)'); end
    end
    
end

if mpi_amirank0()
    
    if any(tests > 0)
        disp('	UNIT TESTS FAILED!');
        disp('	ABSOLUTELY DO NOT COMMIT THIS CODE REVISION!');
        if onedev > 0; disp('	SINGLE DEVICE OPERATION: FAILED'); end
        if devx > 0;   disp('	TWO DEVICES, X PARTITION: FAILED'); end
        if devy > 0;   disp('	TWO DEVICES, Y PARTITION: FAILED'); end
        if devz > 0;   disp('	TWO DEVICES, Z PARTITION: FAILED'); end
        
        if any(functests == 0) || (dorad ~= 'y')
            disp('	>>> SOME UNIT TESTS WERE NOT RUN <<<');
            disp('	>>>   FURTHER ERRORS MAY EXIST   <<<');
        end
    else
        disp('	UNIT TESTS PASSED!');
        if any(functests == 0)
            disp('	>>>       SOME UNIT TESTS WERE NOT RUN        <<<');
            if numel(multidev) == 1; disp('	>>>     MULTI-GPU UNIT TESTS WERE NOT RUN     <<<'); end
            disp('	>>> DO NOT COMMIT CODE UNTIL _ALL_ TESTS PASS <<<');
        end
        if dorad ~= 'y'
            disp('  >>>    FREE RADIATION UNIT TEST WAS NOT RUN    <<<');
            disp('  >>> DO NOT COMMIT IF RADIATION CODE IS CHANGED <<<');
        end
    end
    
end

end
