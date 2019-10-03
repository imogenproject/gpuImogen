% Wave advection simulation

%grid = [512 512 1];
grid = [512 256 1];

%--- Initialize test ---%
run             = DustyShockInitializer(grid);
run.iterMax     = 10000;
run.timeMax = 0.25;
run.info        = 'Shock cloud test';
run.notes       = 'Smash a dense bubble into a shockwave!';

%run.image.interval = 100;
%run.image.mass = true;

run.activeSlices.x = false;
run.activeSlices.xy = false;
run.activeSlices.xyz = true;

run.ppSave.dim1 = 100;
if grid(3) == 1
    run.ppSave.dim2 = 10;
    run.ppSave.dim3 = 100;
else
    run.ppSave.dim2 = 100;
    run.ppSave.dim3 = 10;
end

% Hydrogen!
run.rhoPre = .084;
run.PPre   = 100000;

run.dustLoad = .00001;

run.bubbleDensity = .16;
run.bubbleRadius = 10;
run.shockMach = 3;

rp = RealtimePlotter();
  rp.plotmode = 7;
  rp.plotDifference = 0;
  rp.insertPause = 1;
  rp.spawnGUI = 1;
  rp.iterationsPerCall = 10;
  rp.firstCallIteration = 1;
run.peripherals{end+1} = rp;
fm = FlipMethod();
  fm.iniMethod = ENUM.CFD_HLL; 
%  fm.toMethod = ENUM.CFD_HLLC;
%  fm.atstep = -1;
run.peripherals{end+1} = fm;

rp.plotmode = 2;
rp.cut = [256 128 1];
rp.indSubs = [1 1 512;1 1 256;1 1 1];
rp.movieProps(0, 0, 'RTP_');
rp.vectorToPlotprops(1, [1   1   0   4   1   0   1   3   0   1  10   1   8   1   0   0   0]);
rp.vectorToPlotprops(2, [1  10   0   4   1   1   1   3   0   1  10   1   8   1   0   0   0]);


%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    outpath = imogen(IC);
end

