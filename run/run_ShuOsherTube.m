% Run Shu Osher Tube test.

grid = [1024 2 1];
GIS = GlobalIndexSemantics(); GIS.setup(grid);

%--- Initialize test ---%
run             = ShuOsherTubeInitializer(grid);
run.timeMax     = 0.178;
run.iterMax     = round(10*run.timeMax*grid(1)); % This will give steps max ~ 1.2x required

% These are the conditions per the original Shu & Osher paper:
run.lambda 		= 8; % 8 waves in the box
run.mach 		= 3; % Shock propagating at M=3
run.waveAmplitude 	= .2; % Preshock entropy fluctuation of density amplitude 0.2

run.alias       = 'SO_Tube';
run.info        = 'Shu Osher Tube test.';

run.ppSave.dim2 = 5;

%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

