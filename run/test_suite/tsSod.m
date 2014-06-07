function result = testSod(N, direct)

GIS = GlobalIndexSemantics(); GIS.setup([N 16 1]);

%--- Initialize test ---%
run             = SodShockTubeInitializer([N 16 1]);
run.direction   = direct;
run.shockAngle  = 0;
run.iterMax     = 50000;
run.timeMax     = 0.15;

run.alias       = '';
run.info        = 'Sod shock tube test.';
run.notes       = 'Simple axis aligned shock tube test';

run.ppSave.dim2 = 100;

%--- Run tests ---%
icfile = run.saveInitialCondsToFile();
outpath = imogen(icfile);

od = pwd(); cd(outpath);
u = util_LoadWholeFrame('3D_XYZ',1,1); % Load final output
cd(od); result.pathA = od;

T = sum(u.time.history);
X = SodShockSolution(N, T);

sodME(1) = mean(u.mass(:,1)-X.mass');

run.grid = [2*N 16 1];
GIS.setup(run.grid);
icfile = run.saveInitialCondsToFile();
outpath = imogen(icfile);

od = pwd(); cd(outpath);
u = util_LoadWholeFrame('3D_XYZ',1,1); % Load final output
cd(od); result.pathB = od;

T = sum(u.time.history);
X = SodShockSolution(2*N, T);

sodME(2) = mean(u.mass(:,1)-X.mass');

run.grid = [3*N 16 1];
GIS.setup(run.grid);
icfile = run.saveInitialCondsToFile();
outpath = imogen(icfile);

od = pwd(); cd(outpath);
u = util_LoadWholeFrame('3D_XYZ',1,1); % Load final output
cd(od); result.pathC = od;

T = sum(u.time.history);
X = SodShockSolution(3*N, T);

sodME(3) = mean(u.mass(:,1)-X.mass');

result.meanError = sodME

end
