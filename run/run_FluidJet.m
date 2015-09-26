%  Run a fluid jet test.

%--- Initialize test ---%
grid = [512 512 1];
run                 = JetInitializer(grid);
run.iterMax         = 1000;
run.injectorSize    = 15;
run.offset          = [20 256 0];
run.bcMode.x        = 'const';
run.bcMode.y        = 'const';
run.bcMode.z        = 'circ'; % just for some variety
run.direction       = JetInitializer.X;
run.flip            = false;

run.image.interval  = 10;
run.image.mass      = true;
%run.image.speed     = true;

run.info            = 'Fluid jet test in 2D.';
run.notes           = '';

run.activeSlices.xyz = false;
run.ppSave.dim2 = 50;
run.ppSave.dim3 = 50;

run.injectorSize = 9;
run.jetMass = 3;
run.jetMach = 5;

run.pureHydro = 1;

run.useInSituAnalysis = 0;
run.stepsPerInSitu = 20;
        instruct.plotmode = 4;
        instruct.plotDifference = 0;
        instruct.pause = 0;
run.inSituInstructions = instruct;
run.inSituHandle = @RealtimePlotter;


%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    imogen(IC);
end

