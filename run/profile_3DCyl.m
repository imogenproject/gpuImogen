% This runfile is one of the profiling reference files.
% It simulates a global gas disk in cylindrical coordinates,
% with a rotating coordinate frame, exercising 3D cylindrical
% mode kernels and source terms.

% Warning: this one takes a long time to gather performance metrics on...

% Fits on my Quadro M1000M (1GB)...
grid                =[320 200 120];
run                 = KojimaDiskInitializer(grid);
run.iterMax         = 4;
run.edgePadding     = 0.2;
run.pointRadius     = 0.15;
run.radiusRatio     = 0.65;
run.q = 1.8;

run.useZMirror = 1;

run.geomgr.geometryCylindrical(1, 1, 1, 0, 1); % fixme hack

%run.image.interval  = 50;
%run.image.speed     = true;
%run.image.mass      = true;

run.bgDensityCoeff = .0000001;

run.activeSlices.xy  = false;
%run.activeSlices.xz = true;
run.activeSlices.xyz = true;
run.ppSave.dim3 = 25;

run.bcMode.x        = { ENUM.BCMODE_OUTFLOW, ENUM.BCMODE_OUTFLOW };
run.bcMode.y        = ENUM.BCMODE_CIRCULAR;
run.bcMode.z        = ENUM.BCMODE_OUTFLOW; %ENUM.BCMODE_CONSTANT;

run.compositeSourceOrders = [2 4];

run.pureHydro = true;

run.info        = 'Kojima disk simulation';
run.notes       = '';

fm = FlipMethod();
ifm.iniMethod = ENUM.CFD_HLL;
run.peripherals{end+1} = fm;

run.image.parallelUniformColors = true;

run.frameParameters.omega = 1;
run.frameParameters.rotateCenter = [0 0 0];

%--- Run tests ---%
if (true) %Primary test
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

