% Run a Noh shock tube test.

%--- Initialize test ---%
grid = [1024 1 1];
run  = NohTubeInitializer(grid);

run.rho0 = 1;
run.r0 = -0.8;
run.v0 = -1;
run.M0 = 3;

run.cfl = .7;
NT = NohTubeExactPlanar(run.rho0, 1, run.M0);
run.timeMax = NT.shockTimeGivenPosition(run.r0, -run.r0);;
run.iterMax     = 10*run.timeMax*grid(1)/run.cfl; % This will give steps max ~ 1.2x required

run.alias       = '';
run.info        = 'Noh shock tube test.';
run.notes       = 'Simple axis aligned shock tube test';

run.ppSave.dim3 = 100;

run.useHalfspace([1 0 0]);

rp = RealtimePlotter();
  rp.plotmode = 4;
  rp.plotDifference = 0;
  rp.insertPause = 1;
  rp.firstCallIteration = 0;
  rp.iterationsPerCall = 1;
  rp.spawnGUI = 1;
run.peripherals{end+1} = rp;


fm = FlipMethod();
fm.iniMethod = ENUM.CFD_HLLC;
run.peripherals{end+1} = fm;

run.saveFormat = ENUM.FORMAT_MAT;

%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

