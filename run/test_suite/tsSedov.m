function result = tsSedov(iniResolution, multiples, prettyPictures, methodPicker)

if nargin < 3
    prettyPictures = 0;
end

grid = iniResolution;

%--- Initialize test ---%
% Default ST blast desposits E=1 into a region 3.5 cells in radius
run         = SedovTaylorBlastWaveInitializer(grid);

run.autoEndtime = 1; % Automatically run until Rblast = 0.45

if grid(3) > 1
    run.mirrordims = [1 1 1];
    run.depositRadiusCells(1.5);
else
    run.mirrordims = [1 1 0];
    run.depositRadiusCells(sqrt(2.5));
end

run.alias   = 'SEDOV_ts';
run.info    = 'Sedov-Taylor blast wave convergence test.';
run.notes   = 'Eblast=1, box diameter = [1 1 1], Rend = 0.45';

run.activeSlices.xy = false;
run.ppSave.dim3      = 2;

result.paths = {};
result.times = [];
result.rhoL1 = [];
result.rhoL2 = [];
result.velL1 = [];
result.velL2 = [];
result.pressL1 = [];
result.pressL2 = [];

ydim = [];

fm = FlipMethod();
  fm.iniMethod = ENUM.CFD_HLLC;
  fm.toMethod = ENUM.CFD_HLL;
  fm.atstep = 20;
run.peripherals{end+1} = fm;

if prettyPictures
    rp = RealtimePlotter();
    rp.plotmode = 4;
    rp.plotDifference = 0;
    rp.insertPause = 0;
    rp.firstCallIteration = 1;
    rp.iterationsPerCall = 25;
    run.peripherals{end+1} = rp;
end
if nargin == 4
    run.peripherals{end+1} = methodPicker;
end


%--- Run tests ---%
for N = 1:numel(multiples)
    grid = iniResolution*multiples(N);
    if iniResolution(2) <= 2; grid(2) = iniResolution(2); end % Keep 1D, 1D
    if iniResolution(3) == 1; grid(3) = 1; end % keep 2D, 2D

    run.geomgr.setup(grid);
    run.iterMax = 100 * max(grid); % Safely make sure that saving is set by time, not iteration
    icfile      = run.saveInitialCondsToFile();
    outdir      = imogen(icfile);

    if N > 1; run.autoEndtime = 0; end % Only run this once to avoid simulation length changing underneath us

    enforceConsistentView(outdir);
    status      = analyzeSedovTaylor(outdir, 1);

    result.paths{end+1} = outdir;

    % Take times from the first run
    if N == 1
        result.times = status.time;
        ydim = numel(status.rhoL1);
    end
    
    if numel(status.rhoL1) < ydim
        warning('\nCurrent ST convergence test run stored at %s was expected to save %I 3D data frames.\nDirectory actually contains %i\nLikely cause unknown; Run crashed? Extreme low resolution?\nRow padded with zeros to fit.\n', outdir, int32(ydim), int32(numel(status.rhoL1)) );
        status.rhoL1((end+1):ydim) = 0;
        status.rhoL2((end+1):ydim) = 0;
    end

    result.rhoL1(end+1,:) = status.rhoL1(1:ydim);
    result.rhoL2(end+1,:) = status.rhoL2(1:ydim);
    
    result.velL1(end+1,:) = status.velL1(1:ydim);
    result.velL2(end+1,:) = status.velL2(1:ydim);
    
    result.pressL1(end+1,:) = status.pressL1(1:ydim);
    result.pressL2(end+1,:) = status.pressL2(1:ydim);
end

end

