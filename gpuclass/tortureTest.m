function tortureTest(multidev, dorad)
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

disp('#########################');
disp('Standing up test instance');
disp('#########################');

addpath('../mpi');
mpi_init();
disp('Started MPI');
GPU_ctrl('peers',1);
disp('Turned on peer memory access');

disp('#########################');
disp('Stand up successful. Begin testing!');
disp('#########################');

disp('	>>> Using the following devices for multidevice testing <<<');
disp(multidev);

% Check that the basic stuff works first
x = GPUManager.getInstance();

basicunits = basicTest(multidev);

disp('#########################');

if basicunits > 0;
    error('Fatal: Unit tests of basic functionality indicate failure. Aborting further tests.');
end

disp('Horray, we got this far! That means basic upload/download is working (or at least not');
disp('manifestly defective). Proceeding to fuzz the following routines which have unit tests:');

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
    disp('>>> WARNING: Only one device indicated. Will not perform multi-device fuzzing tests.');
else
    functests = [1 1 1 1];
end

names = {'cudaArrayAtomic', 'cudaArrayRotateB', 'cudaSoundspeed', 'directionalMaxFinder', 'freezeAndPtot', 'cudaSourceScalarPotential', 'cudaSourceRotatingFrame'};
if nargin < 2; 
    dorad = input('Test cudaFreeRadiation? This will take longer than the rest combined (y/n): ', 's');
end

if dorad == 'y'; names{end+1} = 'cudaFreeRadiation'; end

for N = 1:numel(names); disp(['	' names{N}]); end
disp('NOTE: setup & data upload times dwarf execution times here; No speedup will be observed.');
disp('#########################');

randSeed = 5418;
res2d = [2048 2048 1];
res3d = [192 192 192];
nTests = 15;

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
        disp('	>>> DO NOT COMMIT CODE UNTIL _ALL_ TESTS PASS <<<');
    end
    if dorad ~= 'y'
        disp('  >>>    FREE RADIATION UNIT TEST WAS NOT RUN    <<<');
        disp('  >>> DO NOT COMMIT IF RADIATION CODE IS CHANGED <<<');
    end
end


end
