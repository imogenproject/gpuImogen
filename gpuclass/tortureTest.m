
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

% If cuda-gdb'ed
if 0;
  multidev = [0 1];
  disp('	>>>  using devices [0 1] for multidevice tests  <<<');
% If normal
else
  multidev = [0 2];
  disp('	>>>  using devices [0 2] for multidevice tests  <<<');
  disp('	>>> If unexpected fail, check device visibility <<<');
end

% Check that the basic stuff works first
x = GPUManager.getInstance();

basicunits = basicTest(multidev);

disp('#########################');

if basicunits > 0;
    error('Fatal: Unit tests of basic functionality indicate failure. Aborting further tests.');
end

disp('Fundamental functionality is working (or at least not spectacularly malfunctioning).');
disp('Fuzzing the following routines which have computable exactly correct answers:');

% Kernels used by Imogen:
% TIME_MANAGER	cudaSoundspeed directionalMaxFinder
% 50% coverage by unit tests
% FLUX		
% 	ARRAY_INDEX_EXCHANGE:	cudaArrayRotateB
%	RELAXING_FLUID:		freezeAndPtot, cudaFluidStep
%	cudahalo exchange: 	oh_god_why.jpg, it's a block of custom shit-code all its own
% 66% coverage by unit tests
% SOURCE
%	cudaSourceRotatingFrame, cudaAccretingStar, cudaSourceScalarPotential, cudaFreeRadiation
% 25% coverage by unit tests (potential test works)

%names = {'cudaArrayAtomic', 'cudaArrayRotateB', 'cudaSoundspeed', 'directionalMaxFinder', 'freezeAndPtot', 'cudaSourceScalarPotential', 'cudaSourceRotatingFrame'};
names = {'cudaArrayAtomic', 'cudaArrayRotateB', 'cudaSoundspeed', 'directionalMaxFinder', 'freezeAndPtot', 'cudaSourceScalarPotential'};
for N = 1:numel(names); disp(['	' names{N}]); end
disp('NOTE: setup & data upload times dwarf execution times here; No speedup will be observed.');
disp('#########################');

randSeed = 5418;
res2d = [2048 2048 1];
res3d = [192 192 192];
nTests = 25;

% Test single-device operation
disp('==================== Testing on one GPU');
x.init([0], 3, 1); rng(randSeed);
onedev = unitTest(nTests, res2d, nTests, res3d, names);

% Test two partitions on different devices
disp('==================== Testing two GPUs, X partitioning');
x.init(multidev, 3, 1); rng(randSeed);
devx = unitTest(nTests, res2d, nTests, res3d, names);
disp('==================== Testing two GPUs, Y partitioning');
x.init(multidev, 3, 2); rng(randSeed);
devy = unitTest(nTests, res2d, nTests, res3d, names);
disp('==================== Testing two GPUs, Z partitioning');
x.init(multidev, 3, 3); rng(randSeed);
devz = unitTest(nTests, res2d, nTests, res3d, names);

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
else
    disp('	UNIT TESTS PASSED!');
    disp('	HORRAY NOT SUCKING!');
end


