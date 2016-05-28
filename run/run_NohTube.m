% Run Sod shock tube test.

%--- Initialize test ---%
grid = [512 512 1];
run             = NohTubeInitializer(grid);

run.rho0 = 1;
run.r0 = 0.2;
run.t0 = 1; % positive time solution
run.v0 = -1;
run.M0 = 10;

run.cfl = .4;
run.iterMax     = 3*run.timeMax*grid(1)/run.cfl; % This will give steps max ~ 1.2x required

run.bcMode = ENUM.BCMODE_CONST;

run.alias       = '';
run.info        = 'Sod shock tube test.';
run.notes       = 'Simple axis aligned shock tube test';

run.ppSave.dim3 = 100;

rp = RealtimePlotter();
  rp.plotmode = 4;
  rp.plotDifference = 0;
  rp.insertPause = 0;
  rp.firstCallIteration = 1;
  rp.iterationsPerCall = 20;
run.peripherals{end+1} = rp;

fm = FlipMethod();
fm.iniMethod = 1; % hll
run.peripherals{end+1} = fm;

run.saveFormat = ENUM.FORMAT_MAT;

%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

