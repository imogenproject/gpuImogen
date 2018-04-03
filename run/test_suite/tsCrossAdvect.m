function result = tsCrossAdvect(wavetype, grid, Nmax, B0, V0, prettyPictures, methodPicker)

if nargin < 6
    prettyPictures = 0;
end

run         = AdvectionInitializer(grid);
run.iterMax = 499999;
run.info    = 'Advection test.';
run.notes   = 'Automated test suite: Advection';

%run.image.interval = 100;
%run.image.mass = true;

run.activeSlices.x = false;
run.activeSlices.xy = false;
run.activeSlices.xyz = true;

run.ppSave.dim1 = 100;
run.ppSave.dim2 = 100;

% Set a background speed at which the fluid is advected to 0
run.backgroundMach = V0;

% Set the type of wave to be run.
% One of 'entropy', 'sound', 'alfven', 'sloww ma', 'fast ma'
% The MHD waves require a B to be set; Setting one is optional for the Entropy wave.
% Any nonzero B will automatically activate magnetic fluxing, and turn 'sonic' into 'fast ma'
% as the fast branch merges with the sonic branch of the dispersion relation
run.waveType = wavetype;
run.amplitude = .05;
run.waveLinearity(0);
run.waveStationarity(0);
% FWIW an amplitude of .01 corresponds to a roughly 154dB sound in air

run.backgroundB = B0;
run.ppSave.dim3 =  100;

if prettyPictures
    rp = RealtimePlotter();
    rp.plotmode = 1;
    if grid(2) > 3; rp.plotmode = 4; end
    rp.plotDifference = 0;
    rp.insertPause = 0;
    rp.firstCallIteration = 1;
    rp.iterationsPerCall = 10;
    rp.spawnGUI = 1;
    run.peripherals{end+1} = rp;
end
if nargin == 7
    run.peripherals{end+1} = methodPicker;
end

result.firstGrid = grid;
result.paths = cell(Nmax+1);

result.L1_prefactor = zeros(Nmax+1);
result.L2_prefactor = zeros(Nmax+1);

totalsims = prod(Nmax+1);
simct = 0;
run.wavenumber = [1 0 0];
run.forCriticalTimes(0.95);
cyc = run.cycles / norm(Nmax);

for nx = 0:Nmax(1)
for ny = 0:Nmax(2)
for nz = 0:Nmax(3)
    if nx+ny+nz == 0; continue; end % zero wavenumber is meaningless

    run.wavenumber = [nx ny nz];
    % this should be the same actual time (i.e. # of steps) every time
    run.cycles = cyc*norm([nx ny nz]);

    wn = int32(run.wavenumber);
    run.alias = sprintf('ADVECT_XY_testsuite_N%i_%i_%i', wn(1), wn(2), wn(3));

    % Run simulation at present resolution & store results path
    IC = run.saveInitialCondsToStructure();
    outpath = imogen(IC);

    enforceConsistentView(outpath); 
    A = SonicAdvectionAnalysis(outpath, 1);
    result.paths{nx+1, ny+1, nz+1} = outpath;

    % Store error norms
    result.L1_prefactor(nx+1, ny+1, nz+1) = A.rhoL1(end);
    result.L2_prefactor(nx+1, ny+1, nz+1) = A.rhoL2(end);

    simct = simct + 1;
    if mpi_amirank0(); fprintf('Done running simulation %i/%i!\n', int32(simct), int32(totalsims)); end
end
end
end

if numel(B0) < 3; B0 = [1 1 1]*B0(1); end % Prevent descriptor sprintf from barfing
if numel(V0) < 3; V0 = [1 0 0]*V0(1); end

[nxgrid nygrid nzgrid] = ndgrid(0:Nmax(1), 0:Nmax(2), 0:Nmax(3));
waveNsqr = nxgrid.^2 + nygrid.^2 + nzgrid.^2;

% prevent 1/0
waveNsqr(1,1,1) = 1;

% scale out asymptotic dependence on N^2 to extract error coefficient
result.L1_prefactor = result.L1_prefactor ./ waveNsqr;
result.L2_prefactor = result.L2_prefactor ./ waveNsqr;

result.mu1 = mean(result.L1_prefactor(:));
result.mu2 = mean(result.L2_prefactor(:));
result.sigma1 = std(result.L1_prefactor(:));
result.sigma2 = std(result.L2_prefactor(:));

end
