function tortureTest(multidev, dorad, nTests)
% tortureTest([device list], 'y'/'n' to radiative cooling test)
% > device list: Set of integers naming GPUs to use, as enumerated by GPU_ctrl('info')
% > dorad: If == 'y' tests cudaFreeRadiation

if nargin < 1
    multidev = [0];
    disp('>>> WARNING: No input array of device IDs given; Defaulting to [0] and disabling multi-device');
end

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

disp('#########################');
disp('Setting up test environment');
disp('#########################');

addpath('../mpi');
mpi_init();
disp('Started MPI');
GPU_ctrl('peers',1);
disp('Turned on peer memory access');

if numel(multidev) > 1
    disp('	>>> Using the following devices for multidevice testing <<<');
    disp(multidev);
end

disp('#########################');
disp('Running basic funtionality tests');
disp('#########################');


% Check that the basic stuff works first
x = GPUManager.getInstance();

basicunits = basicTest(multidev);
if basicunits > 0;
    error('Fatal problem: Unit tests of basic functionality failed. Aborting further tests.');
end

disp('#########################');

% Kernels used by Imogen:
% TIME_MANAGER	cudaSoundspeed directionalMaxFinder
% 100% coverage by unit tests
% FLUX		
% 	ARRAY_INDEX_EXCHANGE:	cudaArrayRotateB
%	RELAXING_FLUID:		freezeAndPtot, cudaFluidStep, cudaHaloExchange
% 75% coverage by unit tests (no fluid step test)
% SOURCE
%	cudaSourceRotatingFrame, cudaAccretingStar, cudaSourceScalarPotential, cudaFreeRadiation
% 50% coverage by unit tests (accreting star broken)

if numel(multidev) < 2;
    functests = [1 0 0 0];
    disp('>>> WARNING: Only one device indicated for use. Multi-device fuzzing tests will not be done');
else
    functests = [1 1 1 1];
end

names = {'cudaArrayAtomic', 'cudaArrayRotateB', 'cudaSoundspeed', 'directionalMaxFinder', 'freezeAndPtot', 'cudaSourceScalarPotential', 'cudaSourceRotatingFrame'};

if dorad == 'y'; names{end+1} = 'cudaFreeRadiation'; end

for N = 1:numel(names); disp(['	' names{N}]); end
disp('NOTE: setup & data upload times dwarf execution times here; No speedup will be observed.');
disp('#########################');

randSeed = 5418;
% Note these do not DEFINE convenient roundish sizes, actual sizes are randomly chosen up to these
res2d = [2048 2048 1];
res3d = [192 192 192];

% Test single-device operation
if functests(1)
    disp('==================== Testing on one GPU');
    x.init([0], 3, 1); rng(randSeed);
    onedev = unitTest(nTests, res2d, nTests, res3d, names);
else; onedev = 0; end

% Test two partitions on different devices
if functests(2)
    disp('==================== Testing two GPUs, X partitioning');
    x.init(multidev, 3, 1); rng(randSeed);
    devx = unitTest(nTests, res2d, nTests, res3d, names);
else; devx = 0; end

if functests(3)
    disp('==================== Testing two GPUs, Y partitioning');
    x.init(multidev, 3, 2); rng(randSeed);
    devy = unitTest(nTests, res2d, nTests, res3d, names);
else; devy = 0; end

if functests(4)
    disp('==================== Testing two GPUs, Z partitioning');
    x.init(multidev, 3, 3); rng(randSeed);
    devz = unitTest(nTests, res2d, nTests, res3d, names);
else; devz = 0; end

disp('#########################');
disp('RESULTS:')

tests = [onedev devx devy devz];
if any(tests > 0);
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
    if any(functests == 0);
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
