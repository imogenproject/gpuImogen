% Plow fluid past an obstacle, generating a von Karman vortex street.

%--- Initialize test ---%
grid = [16 9 9];
run                 = VortexSheetInitializer(grid);
run.iterMax         = 40;

run.image.interval  = 25;
run.image.mass      = true;
run.activeSlices.xyz = true;
run.info            = 'Vortex Sheet test';
run.notes           = '';
run.ppSave.dim3     = 25;
run.mach            = .8;

%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    imogen(IC);
end

