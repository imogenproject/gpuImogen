% Fire a supersonic fluid jet into a magnetized fluid


%--- Initialize test ---%
grid = [2048 1024 1];
run                 = JetInitializer(grid);
run.mode.magnet     = true;
run.iterMax         = 25;
run.cfl             = 0.35;
run.image.interval  = 10;
run.image.mass      = true;
run.image.mach      = true;
run.info            = 'Magnetic jet test.';
run.notes           = '';

run.bcMode.x = 'circ';
run.bcMode.y = 'circ';

% lol magnetism no work anymore

%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    outdir = imogen(IC);
end

%---------------------------------------------------------------------------------------------------
% Run tests
%----------
if (true) %Parallel Magnetic field 
    ics.jetMags         = [magAmp 0 0];
    ics.backMags = [magAmp 0 0];

    IC = run.saveInitialCondsToStructure();
    outdir = imogen(IC);
end

if (false) %Non-Zero By test
    ics.jetMags         = [0 magAmp 0];
    ics.backMags = [0 magAmp 0];

    IC = run.saveInitialCondsToStructure();
    outdir = imogen(IC);
end

if (false) %Non-Zero Bx By test
    run.jetMags         = [magAmp magAmp 0];
    run.backMags = [magAmp magAmp 0];

    IC = run.saveInitialConditionsToStructure();
    outdir = imogen(IC);
end

