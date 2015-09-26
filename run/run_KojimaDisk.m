%   Run a Kojima disk model.

grid                = [1536 1536 1];
run                 = KojimaDiskInitializer(grid);
run.iterMax         = 100;
run.edgePadding     = 0.2;
run.pointRadius     = 0.15;
run.radiusRatio     = 0.65;
run.q = 1.8;

%run.image.interval  = 50;
%run.image.speed     = true;
%run.image.mass      = true;

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

run.useInSituAnalysis = 0;
run.stepsPerInSitu = 10;
run.inSituHandle = @RealtimePlotter;
run.inSituInstructions.plotmode = 4;


run.image.parallelUniformColors = true;

%--- Run tests ---%
if (true) %Primary test
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

