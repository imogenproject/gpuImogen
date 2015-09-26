% Run the implosion symmetry test.

%--- Initialize test ---%
grid = [512 512 1];
run                 = ImplosionInitializer(grid);
run.iterMax         = 250;

obj.Mcorner         = 0.125;
obj.Pcorner         = 0.14;

run.image.interval  = 100;
run.image.mass      = true;
run.activeSlices.xy = true;
run.info            = 'Implosion symmetry test';
run.notes           = '';
run.ppSave.dim2     = 10;

% The following tracks the asymmetry of the implosion over time

run.useInSituAnalysis = 0;
run.stepsPerInSitu = 20;
run.inSituHandle = @RealtimePlotter;
        instruct.plotmode = 4;
        instruct.plotDifference = 0;
        instruct.pause = 0;
run.inSituInstructions = instruct;


%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    imogen(IC);
end

