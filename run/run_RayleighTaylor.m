%   Run Rayleigh-Taylor instability shock test.

%--- Initialize test ---%
grid = [512 512 1];
run         = RayleighTaylorInitializer(grid);
run.iterMax = 5000;

run.rhoTop           = 2;
run.rhoBottom        = 1;
run.P0               = 2.5;
run.gravity.constant = .1;
run.bcMode.x = ENUM.BCMODE_CIRCULAR;
run.bcMode.y = {ENUM.BCMODE_MIRROR,ENUM.BCMODE_STATIC};
run.bcMode.z = ENUM.BCMODE_CIRCULAR;


run.pertAmplitude = .1;
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

