
disp('######################');
disp('Standing up test instance');
disp('######################');

addpath('../mpi');
mpi_init();
disp('Started MPI');
GPU_ctrl('peers',1);
disp('Turned on peer memory access');

disp('######################');
disp('Stand up successful. Begin testing!');
disp('######################');

% If cuda-gdb'ed
%multidev = [0 1];
% If normal
multidev = [0 2];

% Check that the basic stuff works first
x = GPUManager.getInstance();

basicTest(multidev);

disp('######################');
disp('Fundamental functionality appears to be working correctly.');
disp('Fuzzing the following routines which have computable exactly correct answers:');

names = {'cudaArrayAtomic', 'cudaFreeRadiation', 'cudaFwdAverage', 'cudaFwdDifference', 'cudaSoundspeed', 'directionalMaxFinder', 'freezeAndPtot'};
disp(names);

nTests = 25;
% Test simple single-device operation
disp('----------- Testing on one GPU');
x.init([0], 3, 1);
unitTest(nTests, [1024 1024 1], nTests, [128 128 65], names);

% Test two partitions on different devices
disp('----------- Testing two GPUs, X partitioning');
names = {'cudaArrayAtomic', 'cudaFreeRadiation', 'cudaSoundspeed', 'directionalMaxFinder', 'freezeAndPtot'};
x.init(multidev, 3, 1);
unitTest(nTests, [1024 1024 1], nTests, [128 128 65], names);
disp('----------- Testing two GPUs, Y partitioning');
x.init(multidev, 3, 2);
unitTest(nTests, [1024 1024 1], nTests, [128 128 65], names);
disp('----------- Testing two GPUs, Z partitioning');
x.init(multidev, 3, 3);
unitTest(nTests, [1024 1024 1], nTests, [128 128 65], names);

