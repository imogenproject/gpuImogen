% This runfile is one of the profiling reference files
% It works out fluid dynamics in cylindrical mode,
% and makes use of the gravity and rotating frame sources

grid                =[512 2048 1];
run                 = KojimaDiskInitializer(grid);
run.iterMax         = 5;
run.edgePadding     = 0.2;
run.pointRadius     = 0.15;
run.radiusRatio     = 0.65;
run.q = 1.8;

run.useZMirror = 0;

run.geomgr.geometryCylindrical(1, 1, 1, 0, 1); % fixme hack

%run.image.interval  = 50;
%run.image.speed     = true;
%run.image.mass      = true;

run.bgDensityCoeff = .0001;

run.activeSlices.xy  = false;
%run.activeSlices.xz = true;
run.activeSlices.xyz = true;
run.ppSave.dim3 = 100;

run.bcMode.x        = { ENUM.BCMODE_OUTFLOW, ENUM.BCMODE_OUTFLOW };
run.bcMode.y        = ENUM.BCMODE_CIRCULAR;
run.bcMode.z        = ENUM.BCMODE_CIRCULAR; % irrelevant to 2D_XY

run.pureHydro = true;

run.info        = 'Kojima disk simulation';
run.notes       = '';

rp = RealtimePlotter();
  rp.plotmode = 5;
  rp.plotDifference = 0;
  rp.insertPause = 1;
  rp.forceRedraw = 1;
  rp.iterationsPerCall = 1;
  rp.firstCallIteration = 1;
  rp.spawnGUI = 1;

rp.plotmode = 4;

%run.peripherals{end+1}=rp;

fm = FlipMethod();
fm.iniMethod = ENUM.CFD_HLL;
run.peripherals{end+1} = fm;

run.image.parallelUniformColors = true;

run.frameParameters.omega = 1;
run.frameParameters.rotateCenter = [0 0 0];

%--- Run tests ---%
if (true) %Primary test
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

