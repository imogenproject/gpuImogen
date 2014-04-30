function result = testAdvection(wavetype, grid, N0, B0, V0)

GIS = GlobalIndexSemantics(); GIS.setup(grid);

run         = AdvectionInitializer(grid);
run.iterMax = 99999;
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
run.waveDirection = 1;
run.backgroundMach = V0;

% Set the type of wave to be run.
% One of 'entropy', 'sound', 'alfven', 'slow ma', 'fast ma'
% The MHD waves require a B to be set; Setting one is optional for the Entropy wave.
% Any nonzero B will automatically activate magnetic fluxing, and turn 'sonic' into 'fast ma'
% as the fast branch merges with the sonic branch of the dispersion relation
run.waveType = wavetype;
run.waveAmplitude = .0001;
% FWIW an amplitude of .0001 corresponds to a roughly 114dB sound in air

% number of transverse wave periods in Y and Z directions
run.numWavePeriods = 2;
run.backgroundB = B0;

run.alias = sprintf('ADVECT_N%i_%i_%i',run.waveN(1),run.waveN(2),run.waveN(3));

run.ppSave.dim3 =  100;

run.waveN = N0;
IC = run.saveInitialCondsToStructure();
outpath = imogen(IC);
errOmegaA = AdvectionAutoanalyze(outpath);
result.pathA = outpath;

run.waveN = 2*N0;
IC = run.saveInitialCondsToStructure();
outpath = imogen(IC);
errOmegaB = AdvectionAutoanalyze(outpath);
result.pathB = outpath;

run.waveN = 3*N0;
IC = run.saveInitialCondsToStructure();
outpath = imogen(IC);
errOmegaC = AdvectionAutoanalyze(outpath);
result.pathC = outpath;

erelative = [errOmegaA; errOmegaB; errOmegaC];
hrelative = log([1 2 3]);

fprintf('X-aligned sonic advection test with static background: Error omegas of %g+%gi, %g+%gi, %g+%gi\n', erelative(1,1),erelative(1,2),erelative(2,1),erelative(2,2),erelative(3,1),erelative(3,2));

Ophase = diff(log(abs(erelative(:,1))))./diff(hrelative');
Odamp  = diff(log(abs(erelative(:,2))))./diff(hrelative');

result.order = Ophase + 1i*Odamp;

fprintf('Phase error order: %f, %f\nDamping error order: %f, %f\n', Ophase(1), Ophase(2), Odamp(1), Odamp(2));



end
