function result = tsEinfeldt(N0, gamma, M, doublings)
% result = tsEinfeldt(N0, doublings, M) runs a sequence of Einfeldt tests with
% 2^{0, 1, ..., doublings)*N0 cells, all initialized with -ml = mr = M*cs;
% The normalization rho = P = 1 is used.

if nargin < 4;
    disp('Number of doublings not given; Defaulted to 3.');
    doublings = 3;
end

%--- Initialize test ---%
run             = EinfeldtInitializer([N0 2 1]);
% Run the test until the rarefaction propagates 95% of the way to the edge of
% the grid.
run.timeMax     = .95*.5/((1+M)*sqrt(gamma));
run.iterMax     = 9999;

run.cfl 	= .4;
run.gamma = gamma;

run.rhol	= 1;
run.ml		= -M*sqrt(run.gamma);
run.nl		= 0;
run.el		= 1/(run.gamma-1) + .5*(run.ml^2 + run.nl^2);

run.rhor	= 1;
run.mr		= M*sqrt(run.gamma);
run.nr		= 0;
run.er		= 1/(run.gamma-1) + .5*(run.ml^2+run.nr^2);

%run.useInSituAnalysis = 0;
%run.stepsPerInSitu = 10;
%run.inSituHandle = @RealtimePlotter;

run.alias       = '';
run.info        = 'Einfeldt Strong Rarefaction test.';
run.notes	= '';
run.ppSave.dim3 = 100;

result.N = [];
result.L1 = [];
result.L2 = [];

%--- Run tests ---%
for R = 1:doublings;
    % Set resolution and go
    run.grid = [N0*2^R 2 1];
    icfile = run.saveInitialCondsToFile();
    dirout = imogen(icfile);

    % Access final state
    S = SavefilePortal(dirout);
    S.setFrametype(7);
    f = S.jumpToLastFrame();

    % Generate analytic solution and compute metrics
    T = sum(f.time.history);
    X = ((1:run.grid(1))')/run.grid(1) - .5;

    [rho v P] = einfeldtSolution(X, 1, M*sqrt(gamma), 1, run.gamma, T);

    result.N(end+1) = run.grid(1);
    result.L1(end+1) = mpi_sum(norm(f.mass(:,1) - rho,1)) / run.grid(1);
    result.L2(end+1) = sqrt(mpi_sum(norm(f.mass(:,1) - rho,2).^2))/sqrt(run.grid(1));
end


end
