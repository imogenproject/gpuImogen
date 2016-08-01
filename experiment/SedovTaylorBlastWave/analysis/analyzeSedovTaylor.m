function result = analyzeSedovTaylor(directory, runParallel)

S = SavefilePortal(directory);
S.setFrametype(7);

if nargin < 2; runParallel = 0; end
S.setParallelMode(runParallel);
% initialize to be safe
mpi_init();

result.path = directory;
result.time  = []; 
result.rhoL1 = [];
result.rhoL2 = [];

f = S.nextFrame();

rez = f.parallel.globalDims;
geom = GeometryManager(rez);
geom.geometrySquare(-floor(rez/2) + 0.5, [1 1 1]);

[x, y, z] = geom.ndgridSetXYZ('pos');
R = x.^2+y.^2; if rez(3) > 1; R = R + z.^2; end

R = sqrt(R);
clear x y z;

IC = S.returnInitializer();

rho0 = IC.ini.backgroundDensity;
Eblast = IC.ini.sedovExplosionEnergy;
sedovAlpha = IC.ini.sedovAlphaValue;
% Avoid recalculating the scaling prefactor which makes integral() whine

% Pick an appropriate number of radial cells; Resulting truncation error O(h^3) will be effectively 0.
nRadial = max(size(f.mass));
radii = (0:nRadial)/nRadial;

spatialDimension = 1 + 1*(rez(2) > 2) + 1*(rez(3) > 1);

for N = 1:S.numFrames()
    [rho, vradial, P] = SedovSolver.FlowSolution(1, sum(f.time.history), radii, rho0, f.gamma, spatialDimension, sedovAlpha);

    truerho = interp1(radii, rho, R);

    delta = f.mass - truerho;

    if runParallel; delta = geom.withoutHalo(delta); end

    result.time(end+1)  = sum(f.time.history);
    result.rhoL1(end+1) = mpi_sum(norm(delta(:),1)) / mpi_sum(numel(delta));
    result.rhoL2(end+1) = sqrt(mpi_sum(norm(delta(:),2).^2) / mpi_sum(numel(delta)));

    f = S.nextFrame();
end


end
