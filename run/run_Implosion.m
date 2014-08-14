% Run the implosion symmetry test.

%-- Initialize Imogen directory ---%
starterRun();

grid = [512 512 1];
GIS = GlobalIndexSemantics(); GIS.setup(grid);

%--- Initialize test ---%
run                 = ImplosionInitializer(grid);
run.iterMax         = 10000;

run.direction       = ImplosionInitializer.X;
run.image.interval  = 100;
run.image.mass      = true;
run.activeSlices.xy = true;
run.info            = 'Implosion symmetry test';
run.notes           = '';
run.ppSave.dim2     = 10;

% The following tracks the asymmetry of the implosion over time
run.useInSituAnalysis = 1;
run.inSituHandle = @ImplosionAnalyzer.getInstance;
run.stepsPerInSitu = 100;

%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    imogen(IC);
end

