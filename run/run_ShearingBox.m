%   Run a Kojima disk model.

grid                = [384 1 128];
run                 = ShearingBoxInitializer(grid);
run.iterMax         = 1000;

run.Mstar           = 2e30;
run.Rstar           = 1e9;

% 5 to 20 AU
run.innerRadius = 150e9*8;
run.outerRadius = 150e9*16;

run.Sigma = 1000; % 100g/cm^2 = 1000kg/m^2
run.densityExponent = -1; % steady state accretion disk
run.temperatureExponent = -0.5; % P_solar ~ r^-2 = P_bbody ~ T^4 -> T ~ r^-0.5

run.geomgr.geometryCylindrical(1, 1, 1, 0, 1); % just to pop it into cylindrical geometry mode


%run.image.interval  = 50;
%run.image.speed     = true;
%run.image.mass      = true;

if grid(3) == 1
    run.activeSlices.xy  = true;
else
    run.activeSlices.xyz = true;
end
run.ppSave.dim3 = 100;

run.VTOSettings = [1 1] / (31e6*10);
run.checkpointSteps = 100;

run.info        = 'Shearing dusty box simulation';
run.notes       = '';

rp = RealtimePlotter();
  rp.plotmode = 7;
  rp.plotDifference = 0;
  rp.insertPause = 1;
  rp.forceRedraw = 1;
  rp.iterationsPerCall = 1;
  rp.firstCallIteration = 1;
  rp.spawnGUI = 1;
run.peripherals{end+1} = rp;

fm = FlipMethod();
fm.iniMethod = ENUM.CFD_HLL;
run.peripherals{end+1} = fm;

run.image.parallelUniformColors = true;

%run.frameParameters.omega = 2.1715;
%run.frameParameters.rotateCenter = [256.5 256.5 0];

%--- Run tests ---%
if (true) %Primary test
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

