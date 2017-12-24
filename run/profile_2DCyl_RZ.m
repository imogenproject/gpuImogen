% This runfile is one of the profiling reference files
% It runs a cylindrical coordinates simulation as an azimuthal slice
% This exercies the RZ-mode for kernels where it matters and the change
% in grid values often reveals performance defects.
% CFD and rotating frame are exercised in cylindrical coordinates

grid                =[1280 1 480];
run                 = KojimaDiskInitializer(grid);
run.iterMax         = 5;
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
run.ppSave.dim3 = 100;

run.bcMode.x        = { ENUM.BCMODE_OUTFLOW, ENUM.BCMODE_OUTFLOW };
run.bcMode.y        = ENUM.BCMODE_CIRCULAR;
run.bcMode.z        = ENUM.BCMODE_OUTFLOW; %ENUM.BCMODE_CONSTANT;

run.pureHydro = true;

run.info        = 'Kojima disk simulation';
run.notes       = '';

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

