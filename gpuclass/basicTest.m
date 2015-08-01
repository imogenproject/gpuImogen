function result = basicTest(multidevice, REZ)

x = GPUManager.getInstance();

if nargin == 1; REZ = [449 875]; end

disp('==================== TESTING FUNDAMENTAL UPLOAD/DOWNLOAD/COPY FUNCTIONALITY');
% SINGLE DEVICE UPLOAD/DOWNLOAD
x.init([0], 1, 1);
disp('Testing upload/download/copy to one GPU...');
A = rand(REZ);
Ad = GPU_Type(A);
B = max(max(abs(Ad.array - A)));
if B ~= 0;
    disp('Scalar GPU ul/dl: Returned data failed to be exactly equal to original: Failure.');
    result = 1; return
else; disp('	UL/DL successful!'); end

% SINGLE DEVICE DEEP COPY
Cd = GPU_clone(Ad);
B = max(max(abs(GPU_download(Cd) - Ad.array)));
if B ~= 0;
    disp('GPU_clone data was not identical to original data after download: Failure.');
    result = 1; return;
else; disp('	Deep copy successful!'); end


%%%%%%%%%%%%%%% MULTIDEVICE, X PARTITION, UL/DL
x.init(multidevice, 3, 1);
disp('Testing upload/download to two GPUs, partition dir = X');
A = rand(REZ);
Ad = GPU_Type(A);
theta = Ad.array;
B = max(max(abs(theta - A)));
if B ~= 0;
    disp('Multi-device ul/dl: Returned data failed to be exactly equal to original: Failure.');
    result = 1; return;
else; disp('	UL/DL successful!'); end

% MULTIDEVICE DEEP COPY, X PARTITION
Cd = GPU_clone(Ad);
B = max(max(abs(GPU_download(Cd) - Ad.array)));
if B ~= 0; 
    disp('GPU_clone does not work.');
    result = 1; return;
else; disp('	Deep copy creation successful!'); end
GPU_free(Cd);

% MULTIDEVICE ARRAY MEMCPY, X PARTITION
Cd = GPU_Type(rand(REZ) );
GPU_copy(Cd.GPU_MemPtr, Ad.GPU_MemPtr);
B = max(max(abs(GPU_download(Cd) - Ad.array)));
if B ~= 0;
    disp('GPU_copy does not work.');
    result = 1; return;
else; disp('	Overwrite copy successful!'); end

%%%%%%%%%%%%%%5% MULTIDEVICE, Y PARTITION, UL/DL
x.init(multidevice, 3, 2);
disp('Testing upload/download to two GPUs, partition dir = Y');
A = rand(REZ);
Ad = GPU_Type(A);
theta = Ad.array;
B = max(max(abs(theta - A)));
if B ~= 0;
    disp('Multi-device ul/dl: Returned data failed to be exactly equal to original: Failure.');
    result = 1; return;
else; disp('	UL/DL successful!'); end

% MULTIDEVICE DEEP COPY, Y PARTITION
Cd = GPU_clone(Ad);
B = max(max(abs(GPU_download(Cd) - Ad.array)));
if B ~= 0;
    disp('GPU_clone does not work.');
    result = 1; return;
else; disp('	Deep copy creation successful!'); end
GPU_free(Cd);

% MULTIDEVICE ARRAY MEMCPY, Y PARTITION
Cd = GPU_Type(rand(REZ) );
GPU_copy(Cd.GPU_MemPtr, Ad.GPU_MemPtr);
B = max(max(abs(GPU_download(Cd) - Ad.array)));
if B ~= 0;
    disp('GPU_copy does not work.');
    result = 1; return;
else; disp('	Overwrite copy successful!'); end

%%%%%%%%%%%%%%% MULTIDEVICE, Z PARTITION, UL/DL
x.init(multidevice, 3, 3);
disp('Testing upload/download to two GPUs, partition dir = Z');
A = rand(REZ);
Ad = GPU_Type(A);
theta = Ad.array;
B = max(max(abs(theta - A)));
if B ~= 0;
    disp('Multi-device ul/dl: Returned data failed to be exactly equal to original: Failure.');
    result = 1; return;
else; disp('	UL/DL successful!'); end

% MULTIDEVICE DEEP COPY CREATION
Cd = GPU_clone(Ad);
B = max(max(abs(GPU_download(Cd) - Ad.array)));
if B ~= 0;
    disp('GPU_clone does not work.');
    result = 1; return;
else; disp('	Deep copy creation successful!'); end
GPU_free(Cd);

% MULTIDEVICE ARRAY MEMCPY, Z PARTITION
Cd = GPU_Type(rand(REZ) );
GPU_copy(Cd.GPU_MemPtr, Ad.GPU_MemPtr);
B = max(max(abs(GPU_download(Cd) - Ad.array)));
if B ~= 0;
    disp('GPU_copy does not work.');
    result = 1; return;
else; disp('	Overwrite copy successful!'); end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('==================== TESTING SLAB FUNCTIONALITY');

disp('Attempting to extend array to have three slabs...');
Ad.createSlabs(3);

disp('Accessing slab 1...');
Btag = GPU_Type(GPU_getslab(Ad, 1));

disp('Setting slab value...');
B = rand(REZ);
Btag.array = B;

disp('Downloading slab...');
diff = max(max(max(abs(Btag.array - B))));
if diff ~= 0;
    disp('Transfer to/from slab failed!'); 
    result = 1; return;
else; disp('Successful!'); end

result = 0;

end

