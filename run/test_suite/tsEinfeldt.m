function result = tsEinfeldt(N0, gamma, M, doublings, prettyPictures, methodPicker)
% result = tsEinfeldt(N0, doublings, M) runs a sequence of Einfeldt tests with
% 2^{0, 1, ..., doublings)*N0 cells, all initialized with -ml = mr = M*cs;
% The normalization rho = P = 1 is used.

if nargin < 4;
    disp('Number of doublings not given; Defaulted to 3.');
    doublings = 3;
end
if nargin < 5
    prettyPictures = 0;
end

%--- Initialize test ---%
grid = [N0 1 1 ];
run             = RiemannProblemInitializer(grid);
% Run the test until the rarefaction propagates 95% of the way to the edge of
% the grid.
run.timeMax     = .92*.5/((1+M)*sqrt(gamma));
run.iterMax     = 99999;

run.cfl 	    = .75;

run.setupEinfeldt(M, gamma)
run.bcMode.x    = ENUM.BCMODE_CONSTANT;

%run.useInSituAnalysis = 0;
%run.stepsPerInSitu = 10;
%run.inSituHandle = @RealtimePlotter;

run.alias       = '';
run.info        = 'Einfeldt Strong Rarefaction test.';
run.notes	    = '';
run.ppSave.dim3 = 100;

%fm = FlipMethod();
%  fm.iniMethod = 2; % hllc
%run.peripherals{end+1} = fm;

if prettyPictures
    rp = RealtimePlotter();
    rp.plotmode = 1;
    rp.plotDifference = 0;
    rp.insertPause = 0;
    rp.firstCallIteration = 1;
    rp.iterationsPerCall = 25;
    run.peripherals{end+1} = rp;
end
if nargin == 6
    run.peripherals{end+1} = methodPicker;
end

result.N = [];
result.L1 = [];
result.L2 = [];
result.paths = {};

%--- Run tests ---%
for R = 1:doublings;
    % Set resolution and go
    grid(1) = N0 * 2^R;
    run.geomgr.setup(grid);
    icfile = run.saveInitialCondsToFile();
    dirout = imogen(icfile);
    enforceConsistentView(dirout);

    result.paths{end+1} = dirout;

    % Access final state
    S = SavefilePortal(dirout);
    S.setFrametype(7);
    f = S.jumpToLastFrame();

    % Generate analytic solution and compute metrics
    % NOTE: use run.geometry.localXposition?
    T = sum(f.time.history);
    X = ((1:run.geomgr.globalDomainRez(1))' + .5)/run.geomgr.globalDomainRez(1) - .5;

    [rho, v, P] = einfeldtSolution(X, 1, M*sqrt(gamma), 1, run.gamma, T);

    result.N(end+1) = run.geomgr.globalDomainRez(1);
    result.L1(end+1) = mpi_sum(norm(f.mass(:,1) - rho,1)) / run.geomgr.globalDomainRez(1);
    result.L2(end+1) = sqrt(mpi_sum(norm(f.mass(:,1) - rho,2).^2))/sqrt(run.geomgr.globalDomainRez(1));
end


end
