%   Run a Kojima disk model.

grid                = [384 256 1];
run                 = KojimaDiskInitializer(grid);
run.iterMax         = 10000;
run.edgePadding     = 0.4;
run.pointRadius     = 0.15;
run.radiusRatio     = 0.65;
run.q = 1.8;

run.geomgr.geometryCylindrical(1, 1, 1, 0, 1); % fixme hack

%run.image.interval  = 50;
%run.image.speed     = true;
%run.image.mass      = true;

run.bgDensityCoeff = .0001;

run.activeSlices.xy  = false;
%run.activeSlices.xz = true;
run.activeSlices.xyz = true;
run.ppSave.dim3 = 100;

run.bcMode.x        = ENUM.BCMODE_CONSTANT;
run.bcMode.y        = ENUM.BCMODE_CONSTANT;
run.bcMode.z        = ENUM.BCMODE_CONSTANT;

run.pureHydro = true;
run.cfl = .7;

run.info        = 'Kojima disk simulation';
run.notes       = '';

rp = RealtimePlotter();
  rp.plotmode = 7;
  rp.plotDifference = 0;
  rp.insertPause = 1;
  rp.forceRedraw = 1;
  rp.iterationsPerCall = 10;
  rp.firstCallIteration = 1;
run.peripherals{end+1} = rp;

fm = FlipMethod();
fm.iniMethod = 1; % hll
run.peripherals{end+1} = fm;

run.image.parallelUniformColors = true;

%run.frameParameters.omega = 2.1715;
%run.frameParameters.rotateCenter = [256.5 256.5 0];

%--- Run tests ---%
if (true) %Primary test
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

