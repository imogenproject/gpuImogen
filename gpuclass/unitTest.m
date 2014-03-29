function result = unitTest(n2d, max2d, n3d, max3d, funcList)

disp('### Beginning GPU functional tests ###');
disp('### Debug-if-error is set.');
dbstop if error

if(nargin < 4)
    disp('unitTest help:');
    disp('unitTest(n2d, max2d, n3d, max3d, funcList):');
    disp('    n2d:   Number of 2 dimensional resolutions to fuzz functions with');
    disp('    max2d: Maximum resolution to use for 2D tests');
    disp('    n3d:   Number of 3 dimensional resolutions to fuzz functions with');
    disp('    max3d: Maximum resolution to use for 3D tests');
    disp('    funcList: {''strings'', ''naming'', ''functions''} or blank to test all.');
    result = 0; return;
end

if(nargin == 4)
    funcList = { 'cudaArrayAtomic', 'cudaArrayRotate', 'freezeAndPtot' };
end

nFuncs = numel(funcList);

D = [2*ones(n2d,1); 3*ones(n3d,1)];
R = [max2d*ones(n2d,1); max3d*ones(n3d,1)];

t0 = tic;

for F = 1:nFuncs;
    outcome = -1;

    targfunc = @noSuchThing; % Default to failure
    if strcmp(funcList{F}, 'cudaArrayAtomic'); targfunc = @testCudaArrayAtomic; end
    if strcmp(funcList{F}, 'cudaArrayRotate'); targfunc = @testCudaArrayRotate; end
    if strcmp(funcList{F}, 'cudaArrayRotate2'); targFunc = @testCuda; end
    if strcmp(funcList{F}, 'cudaFluidTVD'); targFunc = @testCudaFluidTVD; end
    if strcmp(funcList{F}, 'cudaFluidW'); targFunc = @testCudaFluidW; end
    if strcmp(funcList{F}, 'cudaFreeRadiation'); targFunc = @testCudaFreeRadiation; end
    if strcmp(funcList{F}, 'cudaFwdAverage'); targFunc = @testCudaFwdAverage; end
    if strcmp(funcList{F}, 'cudaFwdDifference'); targFunc = @testCudaFwdDifference; end
    if strcmp(funcList{F}, 'cudaMHDKernels'); targFunc = @testCudaMHDKernels; end
    if strcmp(funcList{F}, 'cudaMagPrep'); targFunc = @testCudaMagPrep; end
    if strcmp(funcList{F}, 'cudaMagTVD'); targFunc = @testCudaMagTVD; end
    if strcmp(funcList{F}, 'cudaMagW'); targFunc = @testCudaMagW; end
    if strcmp(funcList{F}, 'cudaShift'); targFunc = @testCudaShift; end
    if strcmp(funcList{F}, 'cudaSoundspeed'); targFunc = @testCudaSoundspeed; end
    if strcmp(funcList{F}, 'cudaSourceAntimach'); targFunc = @testCudaSourceAntimach; end
    if strcmp(funcList{F}, 'cudaSourceRotatingFrame'); targFunc = @testCudaSourceRotatingFrame; end
    if strcmp(funcList{F}, 'cudaSourceScalarPotential'); targFunc = @testCudaSourceScalarPotential; end
    if strcmp(funcList{F}, 'cudaStatics'); targFunc = @testCudaStatics; end
    if strcmp(funcList{F}, 'directionalMaxFinder'); targFunc = @testDirectionalMaxFinder; end
    if strcmp(funcList{F}, 'freezeAndPtot'); targFunc = @testFreezeAndPtot; end
    
    outcome = iterateOnFunction(targfunc, D, R);

    switch outcome;
        case 1:  fprintf('Testing %s failed!\n', funcList{F});
        case 0:  fprintf('Testing %s successful!\n', funcList{F});
        case -1: fprintf('Test for function named %s not implemented\n', funcList{F});
	case -2: fprintf('No function named %s...\n', funcList{F});
    end

end

fprintf('### Tested %i functions in %fsec ###\n', nFuncs, t0 - toc);
end

function outcome = iterateOnFunction(fname, D, R)
    outcome = 0;
    for n = 1:numel(D)
        res = randomResolution(D(n), R(n));
        outcome = fname(res);
        if outcome ~= 0; break; end
    end
end

function R = randomResolution(dim, nmax)
    R = round(nmax*rand(2,1));
    if dim < 3; R(3) = 1; end
end

% Provide a default 'return does-not-exist' target
function failure = noSuchThing(res); failure = 1; end

function outcome = testCudaArrayAtomic(res)
    fail = 0;
    res = randomResolution(D(n), R(n));
    X = rand(res);
    Xg = GPU_Type(X);

    cudaArrayAtomic(Xg, .4, 1);
    X(X < .4) = .4;
    if any(Xg.array(:) ~= X(:) ); fprintf('  !!! Test failed: setting min, res=%ix%ix%in', res(1),res(2), res(3) ); fail = 1; end

    cudaArrayAtomic(Xg, .6, 2);
    X(X > .6) = .6;
    if any(Xg.array(:) ~= X(:) ); fprintf('  !!! Test failed: setting max, res=%ix%ix%in', res(1),res(2), res(3) ); fail = 1; end
  
    X( (X > .4) & (X < .6) ) = NaN;
    Xg.array = X;
    X(isnan(X)) = 0;
    cudaArrayAtomic(Xg, 0, 3);
    if any(Xg.array(:) ~= X(:) ); fprintf('  !!! Test failed: removing NaN, res=%ix%ix%in', res(1),res(2), res(3) ); fail = 1; end
    fprintf('.');

outcome = fail;
end

function fail = testCudaArrayRotate(res)
    fail = 0;
    X = rand(res);
    Xg = GPU_Type(X);
    Yg = GPU_Type(cudaArrayRotate(Xg,2));
    Yg.array(1);
    Xp = []; for z = 1:res(3); Xp(:,:,z) = transpose(X(:,:,z)); end
    if any(any(any(Yg.array ~= Xp)));  disp('  !!! Test failed: XY transpose !!!'); fail = 1; end
    clear Yg;

    if res(3) > 1
        Yg = GPU_Type(cudaArrayRotate(Xg,3));
        Xp = []; for z = 1:res(2); Xp(:,z,:) = squeeze(X(:,z,:))'; end
        if any(any(any(Yg.array ~= Xp))); disp('   !!! Test failed: XZ transpose !!!'); fail = 1; end
     end

end

function fail = testCudaArrayRotate2(res)
fail = -1;
end

function fail = testCudaFluidTVD(res)
fail = -1;
end

function fail = testCudaWStep(res)
    fail = 0;
    [xpos ypos zpos] = ndgrid((1:res(1))*2*pi/res(1), (1:res(2))*2*pi/res(2), (1:res(3))*2*pi/res(3));
    rho = ones(res);
    px = zeros(res) + sin(xpos);
    py = ones(res) + sin(ypos + zpos);
    pz = cos(zpos);
    E  = .5*(px.^2+py.^2+pz.^2)./rho + 2;
    Bx = zeros(res);
    By = zeros(res);
    Bz = zeros(res);

    disp('    Hydro');

    ptot = (2/3)*(E - .5*(px.^2+py.^2+pz.^2)./rho);
    if res(3) == 1
      freeze = max(sqrt((5/3)*ptot./rho) + abs(px./rho),[], 1)';
    else
      freeze = max(sqrt((5/3)*ptot./rho) + abs(px./rho),[], 1);
    end

    rhoD = GPU_Type(rho);
    pxD = GPU_Type(px); pyD = GPU_Type(py); pzD = GPU_Type(pz);
    ED = GPU_Type(E);
    BxD = GPU_Type(Bx); ByD = GPU_Type(By); BzD = GPU_Type(Bz);
    PtotD = GPU_Type(ptot);
    freezespeed = GPU_Type(10*ones([size(rho,2) size(rho,3)]));

    [rhow Ew pxw pyw pzw] = cudaFluidW(rhoD, ED, pxD, pyD, pzD, BxD, ByD, BzD, PtotD, freezespeed, .1, 1);

    % Now to find out what the exact goddamn answer is :/ 
    % test stuff here
end

function fail = testCudaFreeRadiation(res)
fail = -1;
end

function fail = testCudaFwdAverage(res)
fail = -1;
end

function fail = testCudaFwdDifference(res)
fail = -1;
end

function fail = testCudaMHDKernels(res)
fail = -1;
end

function fail = testCudaMagPrep(res)
fail = -1;
end

function fail = testCudaMagTVD(res)
fail = -1;
end

function fail = testCudaMagW(res)
fail = -1;
end

function fail = testCudaPointPotentials(res)
fail = -1;
end

function fail = testCudaShift(res)
fail = -1;
end

function fail = testCudaSoundspeed(res)
fail = -1;
end

function fail = testCudaSourceAntimach(res)
fail = -1;
end

function fail = testCudaSourceRotatingFrame(res)
fail = -1;
end

function fail = testCudaSourceScalarPotential(res)
fail = -1;
end

function fail = testCudaStatics(res)
fail = -1;
end

function fail = testDirectionalMaxFinder(res)
fail = -1;
end


function fail = testFreezeAndPtot(res)
    [xpos ypos zpos] = ndgrid((1:res(1))*2*pi/res(1), (1:res(2))*2*pi/res(2), (1:res(3))*2*pi/res(3));
    % Generate nontrivial conditions
    rho = ones(res);
    px = zeros(res) + sin(xpos);
    py = ones(res) + sin(ypos + zpos);
    pz = cos(zpos);
    E  = .5*(px.^2+py.^2+pz.^2)./rho + 2 + .3*cos(4*xpos) + .1*sin(3*ypos);
    Bx = zeros(res);
    By = zeros(res);
    Bz = zeros(res);

    ptot = (2/3)*(E - .5*(px.^2+py.^2+pz.^2)./rho);
    if res(3) == 1
      freeze = max(sqrt((5/3)*ptot./rho) + abs(px./rho),[], 1)';
    else
      freeze = max(sqrt((5/3)*ptot./rho) + abs(px./rho),[], 1);
    end

    % Translate to GPU vars
    rhoD = GPU_Type(rho);
    pxD = GPU_Type(px); pyD = GPU_Type(py); pzD = GPU_Type(pz);
    ED = GPU_Type(E);
    BxD = GPU_Type(Bx); ByD = GPU_Type(By); BzD = GPU_Type(Bz);

    % Call GPU routine
    [pdev cdev] = freezeAndPtot(rhoD, ED, pxD, pyD, pzD, BxD, ByD, BzD, 5/3, 1);

    pd = GPU_Type(pdev);
    cf = GPU_Type(cdev);
    [ptot(1) pd.array(1) cf.array(1)]
    if max(max(max(abs(pd.array - ptot)))) > 1e-10; disp('   !!! Test failed: P !!!'); fail = 1; end
    if max(max(abs(squeeze(cf.array) - squeeze(freeze)))) > 1e-10; disp('   !!! Test failed: C_f !!!'); fail = 1; end

end

