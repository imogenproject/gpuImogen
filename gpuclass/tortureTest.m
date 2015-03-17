% Check that the basic stuff works first
x = GPUManager.getInstance();

disp('######################');
disp('Testing fundamental GPU functionality');
disp('######################');

basicTest

disp('######################');
disp('Fuzzing routines with exact correct answers');
disp('######################');
nTests = 25;
% Test simple single-device operation
disp('	#######################');
disp('	Testing on one GPU');
disp('	######################');
names = {'cudaArrayAtomic', 'cudaFreeRadiation', 'cudaFwdAverage', 'cudaFwdDifference', 'cudaSoundspeed', 'directionalMaxFinder', 'freezeAndPtot'};
x.init([0], 3, 1);
unitTest(nTests, [1024 1024 1], nTests, [128 128 65], names);

% Test two partitions on the same device (i.e. algorithm w/o P2P access)
disp('	######################');
disp('	Testing two partitions on the same GPU in all 3 partitioning directions');
disp('	######################');
names = {'cudaArrayAtomic', 'cudaFreeRadiation', 'cudaSoundspeed', 'directionalMaxFinder', 'freezeAndPtot'};
x.init([0 0], 3, 1);
unitTest(nTests, [1024 1024 1], nTests, [128 128 65], names);
x.init([0 0], 3, 2);
unitTest(nTests, [1024 1024 1], nTests, [128 128 65], names);
x.init([0 0], 3, 3);
unitTest(nTests, [1024 1024 1], nTests, [128 128 65], names);

% Test two partitions on different devices
disp('	######################');
disp('	Testing two partitions on two GPUs in all 3 partitioning directions');
disp('	######################');
names = {'cudaArrayAtomic', 'cudaFreeRadiation', 'cudaSoundspeed', 'directionalMaxFinder', 'freezeAndPtot'};
x.init([0 2], 3, 1);
unitTest(nTests, [1024 1024 1], nTests, [128 128 65], names);
x.init([0 2], 3, 2);
unitTest(nTests, [1024 1024 1], nTests, [128 128 65], names);
x.init([0 2], 3, 3);
unitTest(nTests, [1024 1024 1], nTests, [128 128 65], names);

