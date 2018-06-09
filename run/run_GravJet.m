
%--- Initialize test ---%
grid = [512 256 1];
run                 = JetInitializer(grid);
run.mode.gravity    = true;
run.direction       = JetInitializer.X;
run.image.interval  = 3;
run.image.mass      = true;
run.image.mach      = true;
run.info            = 'Gravity jet test.';
run.notes           = '';

%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    outpath = imogen(IC);
end

