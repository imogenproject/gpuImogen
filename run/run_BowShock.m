%-- Initialize Imogen directory ---%
starterRun();

%--- Initialize bow shock ---%
grid = [1024 1024 1];
run                 = BowShockInitializer(grid);
run.iterMax         = 2000;
%run.bcMode.z	    = 'circ';

run.bcMode.x = 'const';
run.bcMode.y = 'circ';
run.bcMode.z = 'circ';

run.gpuDeviceNumber = 0;

%--- Adjustable simulation parameters ---%

run.ballXRadius = 1;
run.ballCells = [63.5 63.5 63.5];
run.ballCenter =  [256 512 1];

run.mode.magnet = false;

%run.magX = 1;
%run.magY = 0;

run.preshockRho = 1;
run.preshockP   = 1;
run.blastMach   = 4;

run.ballRho = 1;
run.ballVr = 5;
run.ballXRadius = 1;
run.ballThermalPressure = 2;
run.ballLock = true;

%--- Adjustable output parameters ---%
run.image.interval  = 10;
run.image.mass      = true;
%run.image.speed     = true;
%run.image.pGas      = true;
%run.image.magX      = true;
%run.image.magY = true;

run.activeSlices.xy  = true;
%run.activeSlices.yz  = true;
%run.activeSlices.xyz = true;

run.ppSave.dim2     = 100;
%run.ppSave.dim3     = 20;


run.info            = 'Bow shock test.';
run.notes           = '';

%--- Run tests ---%
if (true)
    [mass, mom, ener, magnet, statics, ini] = run.getInitialConditions();
    IC.mass = mass;
    IC.mom = mom;
    IC.ener = ener;
    IC.magnet = magnet;
    IC.statics = statics;
    IC.ini = ini;
    icfile = [tempname '.mat'];

    save(icfile, 'IC');
    clear IC mass mom ener magnet statics ini run;
    imogen(icfile);
end

enderRun();
