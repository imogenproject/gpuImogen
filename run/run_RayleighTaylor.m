%   Run 3D Corrugation instability shock test.

%-- Initialize Imogen directory ---%
starterRun();

%--- Initialize test ---%
run         = RayleighTaylorInitializer([128 384 1]);
run.info    = 'Rayleigh Taylor instability test';
run.notes   = '';

run.iterMax = 1000;

% run.rhoTop
% run.rhoBottom
% run.P0
% run.B0
% run.gravity.constant

%run.bcMode.gravity.y = 'const';

run.pertAmplitude = .02;
%run.Kx = 4;
run.randomPert = 1;
run.Kz = 0;

run.image.interval = 25;
run.image.mass = true;
%run.image.speed = true;
%run.image.pGas = true;

%run.ppSave.dim2 = 10;

%--- Run tests ---%
if (true) %Primary test
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

