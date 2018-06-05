function result = tsDoubleBlast(iniResolution, doublings, prettyPictures, methodPicker)
%doublings = 7; % will run from 32x1 to 2Kx1
%w0 = 1.5;
% The double blast wave test is a classic test for the shock-capturing codes
% testing the ability to handle very strong (M >> 1) shocks, as well
% as colliding shocks

if nargin < 4
    prettyPictures = 0;
end

grid = iniResolution;
%--- Initialize test ---%
run             = DoubleBlastInitializer(grid);
run.timeMax     = 0.038;
run.iterMax     = 150000;

run.alias       = '';
run.info        = '1D Double Blast Wave test.';
run.notes       = '';

% The original Woodward & Colella (1984) test parameters
run.pRight      = 100;
run.pLeft       = 1000;
run.pMid        = .01;

run.ppSave.dim3 = 25;

run.cfl = 0.85;
run.checkpointSteps = 50;

if prettyPictures
    rp = RealtimePlotter();
    rp.plotmode = 4;
    rp.plotDifference = 0;
    rp.insertPause = 0;
    rp.firstCallIteration = 1;
    rp.iterationsPerCall = 25;
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
    
    grid(1) = grid(1)*2;
end

if mpi_amirank0()

    rhos = cell(doublings,1);

    for N = 1:doublings;
        enforceConsistentView(outdirs{N});
        S = SavefilePortal(outdirs{N});
        S.setParallelMode(0); 

        S.setFrametype('X');

        F = S.jumpToLastFrame();

        rhos{N} = F.mass;
    end

    rhobar = rhos{N};

    for N = (doublings-1):-1:1
        % average 2:1
        rhobar = (rhobar(1:2:end) + rhobar(2:2:end))/2;
    
        result.N(N)  = numel(rhobar);
        result.L1(N) = norm((rhos{N}-rhobar)/numel(rhobar), 1);
        result.L2(N) = norm((rhos{N}-rhobar)/numel(rhobar), 2);
    end

    result.paths = outdirs;
else
    result = [];
end

end
