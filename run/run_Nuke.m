%   Run Rayleigh-Taylor instability shock test.

%--- Initialize test ---%
grid = [256 256 1];
run         = NuclearExplosion(grid);
run.iterMax = 15000;

% run.rho0 = 1;
% run.rho1 = -.1;
run.gravity.constant = .1;
run.bcMode.x = ENUM.BCMODE_CONSTANT;
run.bcMode.y = {ENUM.BCMODE_MIRROR,ENUM.BCMODE_STATIC};
run.bcMode.z = ENUM.BCMODE_CONSTANT;

run.rhoMode = run.RHO_GRAD;
run.rho1 = .2;

run.pertAmplitude = .0;
run.randomPert    = 1;
run.Kz            = 0;
run.Kx            = 1;

%run.ppSave.dim2 = 10;
run.image.interval = 25;
run.image.mass     = true;
run.info           = 'Rayleigh Taylor instability test';
run.notes          = '';

%run.B0               = 0;
%run.bcMode.gravity.y = 'const';
%run.image.pGas       = true;
%run.image.speed      = true;

rp = RealtimePlotter();
  rp.plotmode = 4;
  rp.plotDifference = 0;
  rp.insertPause = 1;
  rp.spawnGUI = 1;
  rp.iterationsPerCall = 20;
  rp.firstCallIteration = 1;
run.peripherals{end+1} = rp;

%--- Run tests ---%
if (true) %Primary test
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

