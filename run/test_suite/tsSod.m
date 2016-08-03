function result = tsSod(N, direct, doublings, prettyPictures, methodPicker)

if nargin < 4
    prettyPictures = 0;
end

%--- Initialize test ---%
grid = [N 1 1];
run             = RiemannProblemInitializer(grid);
run.demo_SodTube();

run.iterMax     = 50000;
run.timeMax     = 0.25;

run.cfl         = 0.75;

run.alias       = '';
run.info        = 'Sod shock tube test.';
run.notes       = 'Simple axis aligned shock tube test';

run.ppSave.dim2 = 100;

run.bcMode.x    = ENUM.BCMODE_CONST;

if prettyPictures
    rp = RealtimePlotter();
    rp.plotmode           = 1;
    rp.plotDifference     = 0;
    rp.insertPause        = 0;
    rp.firstCallIteration = 1;
    rp.iterationsPerCall  = 25;
    run.peripherals{end+1}= rp;
end
if nargin == 5
    run.peripherals{end+1} = methodPicker;
end

%--- Run tests ---%

result.L1    = [];
result.L2    = [];
result.res   = [];
result.paths = {};

for p = 1:doublings;
    % Run test at given resolution
    grid(direct) = N*2^(p-1);
    
    run.geomgr.setup(grid);
    icfile           = run.saveInitialCondsToFile();
    outpath          = imogen(icfile);
    enforceConsistentView(outpath);

    % Load last frame
    S = SavefilePortal(outpath);
    S.setFrametype(7);
    u = S.jumpToLastFrame();

    % Compute L_n integral error norms and output
    % FIXME broken in parallel
    T = sum(u.time.history);
    X = SodShockSolution(run.geomgr.globalDomainRez(direct), T);

    result.L2(p)    = sqrt(mpi_sum(norm(u.mass(:,1)-X.mass',2).^2) / mpi_sum(numel(X.mass)) );
    result.L1(p)    = mpi_sum(norm(u.mass(:,1)-X.mass',1)) / mpi_sum(numel(X.mass));
    result.res(p)   = run.geomgr.globalDomainRez(direct);
    result.paths{p} = outpath;

end

% Compute convergence order
dlogh = -log(2);

result.OrderL1 = diff(log(result.L1)) / dlogh;
result.OrderL2 = diff(log(result.L2)) / dlogh;

end
