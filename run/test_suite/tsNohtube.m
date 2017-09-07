function result = tsNohtube(iniResolution, doublings, prettyPictures, methodPicker)
%doublings = 7; % will run from 32x1 to 2Kx1
%w0 = 1.5;
% The double blast wave test is a classic test for the shock-capturing codes
% testing the ability to handle very strong (M >> 1) shocks, as well
% as colliding shocks

if nargin < 4
    prettyPictures = 0;
end

grid = [iniResolution 1 1];
%--- Initialize test ---%
run  = NohTubeInitializer(grid);

run.rho0 = 1;
run.r0 = -0.8;
run.v0 = -1;
run.M0 = 14;

NT = NohTubeExactPlanar(run.rho0, 1, run.M0);

run.cfl = .7;
run.timeMax     = NT.shockTimeGivenPosition(run.r0, -run.r0);
run.iterMax     = 250000;

run.alias       = '';
run.info        = 'Noh shock tube test.';

run.useHalfspace([1 0 0]);

run.ppSave.dim3 = 100;

if prettyPictures
    rp = RealtimePlotter();
    rp.plotmode = 4;
    rp.plotDifference = 0;
    rp.insertPause = 1;
    rp.firstCallIteration = 1;
    rp.iterationsPerCall = 10;
    rp.spawnGUI = 1;
    run.peripherals{end+1} = rp;

end
if nargin == 5
    run.peripherals{end+1} = methodPicker;
end

run.info        = '';
run.notes       = '';

run.image.parallelUniformColors = true;

result.N = [];
result.L1 = [];
result.L2 = [];
result.paths={};

outdirs = cell(doublings,1);

for N = 1:doublings
    % Run test
    disp(['Running at resolution: ',mat2str(grid)]);
    run.geomgr.setup(grid);
    icfile   = run.saveInitialCondsToFile();
    outdirs{N}   = imogen(icfile);

    enforceConsistentView(outdirs{N});
    S = SavefilePortal(outdirs{N});
    S.setFrametype('XYZ');
    
    F = S.jumpToLastFrame();
    rhoSim = F.mass;

    xExact = NT.shockPositionGivenTime(run.r0, sum(F.time.history));
    [X, Y, Z] = run.geomgr.ndgridSetIJK('pos');
    
    rhoExact = NT.solve(1, X, xExact);
    
    delta = rhoSim - rhoExact;
    
    wig = 22;
    result.N(N)  = numel(delta(wig:end));
    result.L1(N) = norm(delta(wig:end)/(numel(delta(wig:end))), 1);
    result.L2(N) = norm(delta(wig:end)/(numel(delta(wig:end))), 2);

    grid(1) = grid(1)*2;
end

result.paths = outdirs;

if mpi_amirank0()
    d0 = pwd();
    cd(outdirs{1});
%    save('./tsCentrifugeResult.mat','result');
    cd(d0);
end

end
