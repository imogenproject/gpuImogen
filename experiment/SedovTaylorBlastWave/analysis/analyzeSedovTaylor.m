function result = analyzeSedovTaylor(directory, runParallel)

S = SavefilePortal(directory);
S.setFrametype(7);

if nargin < 2; runParallel = 0; end
S.setParallelMode(runParallel);

result.path = directory;
result.time  = []; 
result.rhoL1 = [];
result.rhoL2 = [];

f = S.nextFrame();
rez = f.parallel.globalDims;

GIS = GlobalIndexSemantics(); GIS.setup(rez);

[x y z] = GIS.ndgridSetXYZ(floor(rez/2), 1./rez);
R = x.^2+y.^2; if rez(3) > 1; R = R + z.^2; end

R = sqrt(R);
clear x y z;

% HACKS
rho0 = 1;
Eblast = 1;

% Pick an appropriate number of radial cells; Resulting truncation error O(h^3) will be effectively 0.
nRadial = max(size(f.mass));
radii = (0:nRadial)/nRadial;

for N = 1:S.numFrames()
    [rho vradial P] = SedovSolver.FlowSolution(1, sum(f.time.history), radii, rho0, f.gamma, 3);

    truerho = interp1(radii, rho, R);
    
    delta = f.mass - truerho;

    if runParallel; delta = GIS.withoutHalo(delta); end

    result.time(end+1)  = sum(f.time.history);
    result.rhoL1(end+1) = mpi_sum(norm(delta(:),1)) / mpi_sum(numel(delta));
    result.rhoL2(end+1) = sqrt(mpi_sum(norm(delta(:),2).^2) / mpi_sum(numel(delta)));

    f = S.nextFrame();
end


end
