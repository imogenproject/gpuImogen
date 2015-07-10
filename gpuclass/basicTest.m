function result = basicTest(REZ)

x = GPUManager.getInstance();

if nargin == 0; REZ = [449 875]; end


x.init([0], 1, 1);
disp('Testing upload/download to one GPU...');
A = rand(REZ);
Ad = GPU_Type(A);
B = max(max(abs(Ad.array - A)));
if B ~= 0; error('Scalar GPU ul/dl: Returned data failed to be exactly equal to original. Fail.');
    else; disp('Success!'); end

x.init([0 2], 1, 1);
disp('Testing upload/download to two GPUs...');
A = rand(REZ);
Ad = GPU_Type(A);
theta = Ad.array;
B = max(max(abs(theta - A)));
if B ~= 0; error('Multi-device ul/dl: Returned data failed to be exactly equal to original. Fail.');
    else; disp('Success!'); end

disp('Attempting to copy multi-device GPU array into novel pointer (GPU_clone)...');
Cd = GPU_clone(Ad);
B = max(max(abs(GPU_download(Cd) - Ad.array)));
if B ~= 0; error('GPU_clone does not work.');
    else; disp('Success!'); end

GPU_free(Cd);

disp('Attempting to copy multi-device GPU array between existing pointers (GPU_copy)...');
Cd = GPU_Type(rand(REZ) );
GPU_copy(Cd.GPU_MemPtr, Ad.GPU_MemPtr);
B = max(max(abs(GPU_download(Cd) - Ad.array)));
if B ~= 0; error('GPU_copy does not work.'); end

disp('--- Exercising slab functionality ---');

disp('Attempting to extend array to have three slabs...');
Ad.createSlabs(3);

disp('Accessing slab 1...');
Btag = GPU_Type(GPU_getslab(Ad, 1));

disp('Setting slab value...');
B = rand(REZ);
Btag.array = B;

disp('Downloading slab...');
diff = max(max(max(abs(Btag.array - B))));
if diff ~= 0; error('Transfer to/from slab failed!'); 
    else; disp('Successful!'); end

result = 0;

end

