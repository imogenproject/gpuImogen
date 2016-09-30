% Run a Noh shock tube test.

%--- Initialize test ---%
grid = [512 1 1];
run  = NohTubeInitializer(grid);

run.rho0 = 1;
run.r0 = -0.8;
run.v0 = -1;
run.M0 = 3;

run.cfl = .8;
run.timeMax = 1;
run.iterMax     = 3*run.timeMax*grid(1)/run.cfl; % This will give steps max ~ 1.2x required

run.alias       = '';
run.info        = 'Sod shock tube test.';
run.notes       = 'Simple axis aligned shock tube test';

run.ppSave.dim3 = 100;

rp = RealtimePlotter();
  rp.plotmode = 4;
  rp.plotDifference = 0;
  rp.insertPause = 1;
  rp.firstCallIteration = 1;
  rp.iterationsPerCall = 20;
  rp.spawnGUI = 1;
run.peripherals{end+1} = rp;

fm = FlipMethod();
fm.iniMethod = 2; % hll
run.peripherals{end+1} = fm;

run.saveFormat = ENUM.FORMAT_MAT;

%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

