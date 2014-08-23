% Run the Karman Vortex street test.

%-- Initialize Imogen directory ---%
starterRun();

grid = [1024 512 1];
GIS = GlobalIndexSemantics(); GIS.setup(grid);

%--- Initialize test ---%
run                 = KarmanStreetInitializer(grid);
run.iterMax         = 400;

run.image.interval  = 25;
run.image.mass      = true;
run.activeSlices.xy = true;
run.info            = 'Karman Street test';
run.notes           = '';
run.ppSave.dim2     = 25;
run.mach	    = .8;

%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    imogen(IC);
end

