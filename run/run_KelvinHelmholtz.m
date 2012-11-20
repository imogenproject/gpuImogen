% Run a test of the Kelvin-Helmholtz instability test.

%-- Initialize Imogen directory ---%
starterRun();

%--- Initialize test ---%
run                 = KelvinHelmholtzInitializer([512 512 1]);
run.iterMax         = 2000;
run.direction       = KelvinHelmholtzInitializer.X;
run.image.interval	= 25;
run.image.mass		= true;
run.image.mach		= true;
run.activeSlices.xyz = true;
run.info            = 'Kelvin-Helmholtz instability test.';
run.notes           = '';

%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

