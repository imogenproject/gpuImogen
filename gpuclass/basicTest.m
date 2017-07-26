function result = basicTest(multidevice, REZ)
% result = basicTest([device list], [nx ny nz])
% Runs some fundamental unit tests on Imogen GPU functionality

x = GPUManager.getInstance();

if nargin == 1; REZ = [187 330 84]; end

disp('==================== TESTING FUNDAMENTAL UPLOAD/DOWNLOAD/COPY FUNCTIONALITY');
% SINGLE DEVICE UPLOAD/DOWNLOAD
x.init([0], 1, 1);
disp('Testing functionality using GPU zero...');
if runBasicTest(REZ) ~= 0; result = 1; return; end

x.init(multidevice, 3, 1);
disp('Testing upload/download to two GPUs, partition dir = X');
if runBasicTest(REZ) ~= 0; result = 1; return; end

x.init(multidevice, 3, 2);
disp('Testing upload/download to two GPUs, partition dir = Y');
if runBasicTest(REZ) ~= 0; result = 1; return; end

x.init(multidevice, 3, 3);
disp('Testing upload/download to two GPUs, partition dir = Z');
if runBasicTest(REZ) ~= 0; result = 1; return; end

result = 0;
return;

end

%%%%%%% TEST WITH PRE-SET PARTITIONING SCHEME
function succeed = runBasicTest(REZ)

A = rand(REZ);
Ad = GPU_Type(A);
B = max(max(max(abs(Ad.array - A))));
if B ~= 0;
    disp('!!! GPU ul/dl: Returned data failed to be exactly equal to original: Failure.');
    succeed = 1; return
else; disp('	UL/DL successful!'); end

% SINGLE DEVICE DEEP COPY
Cd = GPU_clone(Ad);
B = max(max(max(abs(GPU_download(Cd) - Ad.array))));
if B ~= 0;
    disp('!!! GPU_clone data was not identical to original data after download: Failure.');
    succeed = 1; return;
else; disp('	Deep copy successful!'); end

% Create slabs
Ad.createSlabs(3);
% Access one
Btag = GPU_Type(GPU_getslab(Ad, 1));
% Set directly
B = rand(REZ);
Btag.array = B;
diff = max(abs(Btag.array(:) - B(:)));
if diff ~= 0
    disp('!!! Simple slab set/download failed!');
    succeed = 1; return;
else; disp('	Simple slab set successful!'); end
% Attempt to use combined fetch/copy method
Btag = GPU_Type(GPU_setslab(Ad, 1, B));
diff = max(abs(Btag.array(:) - B(:)));
if diff ~= 0
    disp('!!! GPU_setslab failed!');
    succeed = 1; return;
else; disp('	GPU_setslab via matlab array upload successful!'); end
% Use combined copy/fetch via deep copy from another GPU array
Btag = GPU_Type(GPU_setslab(Ad, 1, Ad));
diff = max(abs(Btag.array(:) - Ad.array(:)));
if diff ~= 0
    disp('!!! GPU_setslab failed!');
    succeed = 1; return;
else; disp('	GPU_setslab via GPU array deepcopy successful!'); end



succeed = 0;

end


