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
    funcList = { 'cudaArrayAtomic', 'cudaArrayRotate', 'freezeAndPtot', 'cudaFwdAverage', 'cudaFwdDifference', 'freezeAndPtot'};
end

nFuncs = numel(funcList);

D = [2*ones(n2d,1); 3*ones(n3d,1)];
R = [ones(n2d,1)*max2d; ones(n3d,1)*max3d];

tic;
for F = 1:nFuncs;
    outcome = -1;

    targfunc = @noSuchThing; % Default to failure
    if strcmp(funcList{F}, 'cudaArrayAtomic'); targfunc = @testCudaArrayAtomic; end
    if strcmp(funcList{F}, 'cudaArrayRotate'); targfunc = @testCudaArrayRotate; end
    if strcmp(funcList{F}, 'cudaArrayRotate2'); targfunc = @testCuda; end
    if strcmp(funcList{F}, 'cudaFluidTVD'); targfunc = @testCudaFluidTVD; end
    if strcmp(funcList{F}, 'cudaFluidW'); targfunc = @testCudaFluidW; end
    if strcmp(funcList{F}, 'cudaFreeRadiation'); targfunc = @testCudaFreeRadiation; end
    if strcmp(funcList{F}, 'cudaFwdAverage'); targfunc = @testCudaFwdAverage; end
    if strcmp(funcList{F}, 'cudaFwdDifference'); targfunc = @testCudaFwdDifference; end
    if strcmp(funcList{F}, 'cudaMHDKernels'); targfunc = @testCudaMHDKernels; end
    if strcmp(funcList{F}, 'cudaMagPrep'); targfunc = @testCudaMagPrep; end
    if strcmp(funcList{F}, 'cudaMagTVD'); targfunc = @testCudaMagTVD; end
    if strcmp(funcList{F}, 'cudaMagW'); targfunc = @testCudaMagW; end
    if strcmp(funcList{F}, 'cudaShift'); targfunc = @testCudaShift; end
    if strcmp(funcList{F}, 'cudaSoundspeed'); targfunc = @testCudaSoundspeed; end
    if strcmp(funcList{F}, 'cudaSourceAntimach'); targfunc = @testCudaSourceAntimach; end
    if strcmp(funcList{F}, 'cudaSourceRotatingFrame'); targfunc = @testCudaSourceRotatingFrame; end
    if strcmp(funcList{F}, 'cudaSourceScalarPotential'); targfunc = @testCudaSourceScalarPotential; end
    if strcmp(funcList{F}, 'cudaStatics'); targfunc = @testCudaStatics; end
    if strcmp(funcList{F}, 'directionalMaxFinder'); targfunc = @testDirectionalMaxFinder; end
    if strcmp(funcList{F}, 'freezeAndPtot'); targfunc = @testFreezeAndPtot; end

    outcome = iterateOnFunction(targfunc, D, R);

    switch outcome
        case 1;  fprintf('Testing %s failed!\n', funcList{F});
        case 0;  fprintf('Testing %s successful!\n', funcList{F});
        case -1; fprintf('Test for function named %s not implemented\n', funcList{F});
	case -2: fprintf('No function named %s...\n', funcList{F});
    end

end

fprintf('### Tested %i functions in %fsec ###\n', nFuncs, toc);
end

function outcome = iterateOnFunction(fname, D, R)
    outcome = 0;
    for n = 1:numel(D)
        res = randomResolution(D(n), R(n));
        outcome = fname(res); fprintf('.');
        if outcome ~= 0; break; end
    end
end

function R = randomResolution(dim, nmax)
    R = round(nmax*rand(1,3));
    R(R < 2) = 2;
    if dim < 3; R(3) = 1; end
end

% Provide a default 'return does-not-exist' target
function failure = noSuchThing(res); failure = -1; end

function outcome = testCudaArrayAtomic(res)
    fail = 0;
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
    rho = 2 - 1.8*rand(res); rhod = GPU_Type(rho);
    px = 2*rand(res); pxd = GPU_Type(px);
    py = 3*rand(res); pyd = GPU_Type(py);
    pz = 2*rand(res); pzd = GPU_Type(pz);
    bx = rand(res);   bxd = GPU_Type(bx);
    by = rand(res);   byd = GPU_Type(by);
    bz = rand(res);   bzd = GPU_Type(bz); % The del.b constraint doesn't matter, this just calculates |B|^2 to subtract it

    T = .5*(px.^2+py.^2+pz.^2) ./ rho;
    B = .5*(bx.^2+by.^2+bz.^2) / 2;

    Ehydro = T + .5+.25*rand(res); Ehydrod = GPU_Type(Ehydro);
    Emhd = Ehydro + B;             Emhdd = GPU_Type(Emhd);

    thtest = .5;

    % Test HD case
    rtest = GPU_Type(cudaFreeRadiation(rhod, pxd, pyd, pzd, Ehydrod, bxd, byd, bzd, [5/3 .5 1 0 1]));
    rtrue = (rho.^(2-thtest)) .* (Ehydro-T).^(thtest);

    max(rtrue(:)-rtest.array(:))

    tau = .01*.5/max(rtest.array(:));

    cudaFreeRadiation(rhod, pxd, pyd, pzd, Ehydrod, bxd, byd, bzd, [5/3 .1 tau .2 1]);
    dE = tau*(rho.^(2-thtest)).*(Ehydro-T).^thtest;
    dE( (Ehydro-T) < .2*rho ) = 0;
    
    max(Ehydro(:) - dE(:) - Ehydrod.array(:))

    % Test MHD case
    rtest = GPU_Type(cudaFreeRadiation(rhod, pxd, pyd, pzd, Emhdd, bxd, byd, bzd, [5/3 .5 1 0 0]));
    rtrue = (rho.^(2-thtest)) .* (Emhd-T-B).^(thtest);

    max(rtrue(:)-rtest.array(:))

    tau = .01*.5/max(rtest.array(:));

    cudaFreeRadiation(rhod, pxd, pyd, pzd, Emhdd, bxd, byd, bzd, [5/3 .1 tau .2 0]);
    dE = tau*(rho.^(2-thtest)).*(Emhd-T-B).^thtest;
    dE( (Emhd-T-B) < .2*rho ) = 0;
    
    max(Emhd(:) - dE(:) - Emhdd.array(:))
     %mexErrMsgTxt("Wrong number of arguments. Expected forms: rate = cudaFreeRadiation(rho, px, py, pz, E, bx, by, bz, [gamma theta beta*dt Tmin isPureHydro]) or cudaFreeRadiation(rho, px, py, pz, E, bx, by , bz, [gamma theta beta*dt Tmin isPureHydro]\n");
end

function fail = testCudaFwdAverage(res)
F = rand(res);

xi = 1:1:res(1);
yi = 1:1:res(2);
zi = 1:1:res(3);
    
G1 = .5*F(xi, yi, zi) + .5*F(circshift(xi',[-1,0,0]), yi, zi);
G2 = .5*F(xi, yi, zi) + .5*F(xi, circshift(yi',[-1,0,0]), zi);
G3 = .5*F(xi, yi, zi) + .5*F(xi, yi, circshift(zi',[-1,0,0]));

%GPU 

FD = GPU_Type(F);
G1D = GPU_Type(cudaFwdAverage(FD, 1));
G2D = GPU_Type(cudaFwdAverage(FD, 2));
G3D = GPU_Type(cudaFwdAverage(FD, 3));

a = max(G1(:)-G1D.array(:));
b = max(G2(:)-G2D.array(:));
c = max(G3(:)-G3D.array(:));
n = [a,b,c];

if max(abs(n)) < 1e-15;
    fail = 0;
else fail = 1;
end;
end

function fail = testCudaFwdDifference(res)
lambda = 0.1; %could be any real number
R = rand(res);
F = rand(res);

xi = 1:1:res(1);
yi = 1:1:res(2);
zi = 1:1:res(3);

delta_x = F(xi, yi, zi) - F(circshift(xi',[-1,0,0]), yi, zi);
delta_y = F(xi, yi, zi) - F(xi, circshift(yi',[-1,0,0]), zi);
delta_z = F(xi, yi, zi) - F(xi, yi, circshift(zi',[-1,0,0]));

Rx = R-lambda*(delta_x);
Ry = R-lambda*(delta_y);
Rz = R-lambda*(delta_z);

%GPU

RD = GPU_Type(R);
FD = GPU_Type(F);

cudaFwdDifference(RD,FD,1,lambda);
a = max(Rx(:)-RD.array(:));
RD.array = R;
cudaFwdDifference(RD,FD,2,lambda);
b = max(Ry(:)-RD.array(:));
RD.array = R;
cudaFwdDifference(RD,FD,3,lambda);
c = max(Rz(:)-RD.array(:));

n = [a,b,c];

if max(abs(n)) < 1e-15;
    fail = 0;
else fail = 1;
end;
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
    [pdev cdev] = freezeAndPtot(rhoD, ED, pxD, pyD, pzD, BxD, ByD, BzD, 5/3, 1, .01);

    pd = GPU_Type(pdev);
    cf = GPU_Type(cdev);
    fail = 0;
    if max(max(max(abs(pd.array - ptot)))) > 1e-10; disp('   !!! Test failed: P !!!'); fail = 1; end
    if max(max(abs(squeeze(cf.array) - squeeze(freeze)))) > 1e-10; disp('   !!! Test failed: C_f !!!'); fail = 1; end

end

