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
result.velL1 = [];
result.velL2 = [];
result.pressL1 = [];
result.pressL2 = [];

f = S.nextFrame();

rez = f.parallel.globalDims;
geom = GeometryManager(rez);
geom.makeBoxSize([1 1 1]);
geom.makeBoxOriginCoord(floor(geom.globalDomainRez/2 + 0.5));

result.N = rez;

[x, y, z] = geom.ndgridSetIJK('pos');
R = x.^2+y.^2; if rez(3) > 1; R = R + z.^2; end

R = sqrt(R);
R(R == 0) = eps; % prevent 1/0 below
%clear x y z;

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
    truev   = interp1(radii, vradial, R);
    trueP   = interp1(radii, P, R);
    
    truevx = truev .* x ./ R;
    deltav = ((f.momX ./ f.mass) - truevx).^2;
    truevy = truev .* y ./ R;
    deltav = deltav + ((f.momY ./ f.mass) - truevy).^2;
    if rez(3) > 1
        truevz = truev .* z ./ R;
        deltav = deltav + ((f.momZ ./ f.mass) - truevz).^2;
    end
    deltav = sqrt(deltav);
    deltaP = trueP - util_DerivedQty(f, 'gaspressure');
    
    deltarho = f.mass - truerho;

    if runParallel; deltarho = geom.withoutHalo(deltarho); end

    NE = mpi_sum(numel(deltarho));
    
    result.time(end+1)  = sum(f.time.history);
    result.rhoL1(end+1) = mpi_sum(norm(deltarho(:),1) / NE);
    result.rhoL2(end+1) = sqrt(mpi_sum(norm(deltarho(:),2).^2) / NE);
    
    result.velL1(end+1) = mpi_sum(norm(deltav(:),1) / NE);
    result.velL2(end+1) = sqrt(mpi_sum(norm(deltav(:),2).^2) / NE);
    
    result.pressL1(end+1) = mpi_sum(norm(deltaP(:),1) / NE);
    result.pressL2(end+1) = sqrt(mpi_sum(norm(deltaP(:),2).^2) / NE);

    f = S.nextFrame();
end


end
