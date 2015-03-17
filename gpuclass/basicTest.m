x = GPUManager.getInstance();

x.init([0], 1, 1);
disp('Testing successful upload/download with device, single GPU');
A = rand([28 319]);
Ad = GPU_Type(A);
B = max(max(abs(Ad.array - A)));
if B ~= 0; error('Scalar GPU ul/dl: Returned data failed to be exactly equal to original. Fail.'); end

x.init([0 0], 1, 1);
disp('Testing successful upload/download with device, two GPUs');
A = rand([28 319]);
Ad = GPU_Type(A);
B = max(max(abs(Ad.array - A)));
if B ~= 0; error('Partitioned GPU ul/dl: Returned data failed to be exactly equal to original. Fail.'); end

x.init([0 2], 1, 1);
disp('Testing successful upload/download with device, two different GPUs');
A = rand([281 319]);
Ad = GPU_Type(A);
B = max(max(abs(Ad.array - A)));
if B ~= 0; error('Multi-device ul/dl: Returned data failed to be exactly equal to original. Fail.'); end

Cd = GPU_clone(Ad);
B = max(max(abs(GPU_download(Cd) - Ad.array)));
if B ~= 0; error('GPU_clone returned bad data.'); end

GPU_free(Cd);

Cd = GPU_Type(zeros([281 319]) );
GPU_copy(Cd.GPU_MemPtr, Ad.GPU_MemPtr);
B = max(max(abs(GPU_download(Cd) - Ad.array)));
if B ~= 0; error('GPU_copy does not work.'); end

Ad.createSlabs(3);

Btag = GPU_getslab(Ad, 1);

