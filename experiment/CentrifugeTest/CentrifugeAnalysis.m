function result = CentrifugeAnalysis(location, runParallel);

S = SavefilePortal(location);

S.setFrametype(7);

if nargin < 2; runParallel = 0; end
S.setParallelMode(runParallel);

equil = S.nextFrame(); % Load 1st 

tval =  [];
l1val = [];
l2val = [];

initset = S.returnInitializer();

geo = GeometryManager(initset.ini.geometry.globalDomainRez);


for N = 2:S.numFrames();
    f = S.nextFrame();

    tval(end+1) = sum(f.time.history);
    delta = f.mass - equil.mass;

    if runParallel; delta = geo.withoutHalo(delta); end

    l1val(end+1) =      mpi_sum(norm(delta(:),1))   / mpi_sum(numel(delta)) ;
    l2val(end+1) = sqrt(mpi_sum(norm(delta(:),2)^2) / mpi_sum(numel(delta)));
end

result.T = tval;
result.L1 = l1val;
result.L2 = l2val;


end
