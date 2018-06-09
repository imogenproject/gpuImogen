function result = basicTest(multidevice, REZ)
% result = basicTest([device list], [nx ny nz])
% Runs some fundamental unit tests on Imogen GPU functionality

x = GPUManager.getInstance();

if nargin == 1; REZ = [187 330 84]; end

dispifz('==================== TESTING FUNDAMENTAL UPLOAD/DOWNLOAD/COPY FUNCTIONALITY');

% SINGLE DEVICE UPLOAD/DOWNLOAD
x.init(multidevice(1), 1, 1);
dispifz('Testing functionality using GPU zero...');
if mpi_any(runBasicTest(REZ) ~= 0); result = 1; return; end

% MULTIPLE DEVICE UPLOAD/DOWNLOAD
if numel(multidevice) > 1
    x.init(multidevice, 3, 1);
    dispifz('Testing upload/download to two GPUs, partition dir = X');
    if mpi_any(runBasicTest(REZ) ~= 0); result = 1; return; end

    x.init(multidevice, 3, 2);
    dispifz('Testing upload/download to two GPUs, partition dir = Y');
    if mpi_any(runBasicTest(REZ) ~= 0); result = 1; return; end

    x.init(multidevice, 3, 3);
    dispifz('Testing upload/download to two GPUs, partition dir = Z');
    if mpi_any(runBasicTest(REZ) ~= 0); result = 1; return; end
else
    dispifz('Only one device to be used: Not testing multi-device operation');
end

result = 0;
return;

end

%%%%%%% TEST WITH PRE-SET PARTITIONING SCHEME
function succeed = runBasicTest(REZ)

A = rand(REZ);
Ad = GPU_Type(A);
B = max(max(max(abs(Ad.array - A))));
if B ~= 0
    disperr('!!! GPU ul/dl: Returned data failed to be exactly equal to original: Failure.');
    succeed = 1; return
else; dispifz('	UL/DL successful!');
end

% SINGLE DEVICE DEEP COPY
Cd = GPU_clone(Ad);
B = max(max(max(abs(GPU_download(Cd) - Ad.array))));
if B ~= 0
    disperr('!!! GPU_clone data was not identical to original data after download: Failure.');
    succeed = 1; return;
else; dispifz('	Deep copy successful!');
end

% Create slabs
Ad.createSlabs(3);
% Access one
Btag = GPU_Type(GPU_getslab(Ad, 1));
% Set directly
B = rand(REZ);
Btag.array = B;
diff = max(abs(Btag.array(:) - B(:)));
if diff ~= 0
    disperr('!!! Simple slab set/download failed!');
    succeed = 1; return;
else; dispifz('	Simple slab set successful!');
end
% Attempt to use combined fetch/copy method
Btag = GPU_Type(GPU_setslab(Ad, 1, B));
diff = max(abs(Btag.array(:) - B(:)));
if diff ~= 0
    disperr('!!! GPU_setslab failed!');
    succeed = 1; return;
else; dispifz('	GPU_setslab via matlab array upload successful!');
end
% Use combined copy/fetch via deep copy from another GPU array
Btag = GPU_Type(GPU_setslab(Ad, 1, Ad));
diff = max(abs(Btag.array(:) - Ad.array(:)));
if diff ~= 0
    disperr('!!! GPU_setslab failed!');
    succeed = 1; return;
else; dispifz('	GPU_setslab via GPU array deepcopy successful!');
end

succeed = 0;

end

function dispifz(x)
b = mpi_basicinfo();

if b(1) == 1 % serial
    disp(x);
else
    if mpi_amirank0(); disp(x); end
end
end

function disperr(x)
b = mpi_basicinfo();

if b(1) == 1 % serial
    disp(x);
else
    if mpi_amirank0(); fprintf('RANK %i: %s', int32(b(2)), x); end
end

end

