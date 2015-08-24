function tortureTest(multidev)

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
%	RELAXING_FLUID:		freezeAndPtot, cudaFluidStep
%	cudahalo exchange: 	oh_god_why.jpg, it's a block of custom shit-code all its own
% 50% coverage by unit tests (no fluid step, halo xchg)
% SOURCE
%	cudaSourceRotatingFrame, cudaAccretingStar, cudaSourceScalarPotential, cudaFreeRadiation
% 50% coverage by unit tests (accreting star broken, free radiation under development)

functests = [1 1 1 1];

names = {'cudaArrayAtomic', 'cudaArrayRotateB', 'cudaSoundspeed', 'directionalMaxFinder', 'freezeAndPtot', 'cudaSourceScalarPotential', 'cudaSourceRotatingFrame'};

for N = 1:numel(names); disp(['	' names{N}]); end
disp('NOTE: setup & data upload times dwarf execution times here; No speedup will be observed.');
disp('#########################');

randSeed = 5418;
res2d = [2048 2048 1];
res3d = [192 192 192];
nTests = 25;

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

    if any(functests == 0);
        disp('	>>> SOME UNIT TESTS WERE NOT RUN <<<');
	disp('	>>>   FURTHER ERRORS MAY EXIST   <<<');
    end
else
    disp('	UNIT TESTS PASSED!');
    if any(functests == 0);
        disp('	>>>       SOME UNIT TESTS WERE NOT RUN        <<<');
        disp('	>>> DO NOT COMMIT CODE UNTIL _ALL_ TESTS PASS <<<');
    end
end


end
