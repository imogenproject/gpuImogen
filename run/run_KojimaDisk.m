%   Run a Kojima disk model.

grid                = [512 512 1];
run                 = KojimaDiskInitializer(grid);
run.iterMax         = 1000;
run.edgePadding     = 0.2;
run.pointRadius     = 0.15;
run.radiusRatio     = 0.65;
run.q = 1.8;

%run.image.interval  = 50;
%run.image.speed     = true;
%run.image.mass      = true;

run.bgDensityCoeff = .01;

run.activeSlices.xy  = false;
%run.activeSlices.xz = true;
run.activeSlices.xyz = true;
run.ppSave.dim3 = 100;

run.bcMode.x        = ENUM.BCMODE_CONST;
run.bcMode.y        = ENUM.BCMODE_CONST;
run.bcMode.z        = ENUM.BCMODE_CONST;

run.pureHydro = true;
run.cfl = .4;

run.info        = 'Kojima disk simulation';
run.notes       = '';

rp = RealtimePlotter();
  rp.plotmode = 7;
  rp.plotDifference = 0;
  rp.insertPause = 1;
  rp.iterationsPerCall =1;
  rp.firstCallIteration = 1;
run.peripherals{end+1} = rp;

run.image.parallelUniformColors = true;

run.frameParameters.omega = 2.1715;
run.frameParameters.rotateCenter = [256.5 256.5 0];

%--- Run tests ---%
if (true) %Primary test
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

