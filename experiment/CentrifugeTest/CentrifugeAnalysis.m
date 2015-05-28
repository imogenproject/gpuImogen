function result = CentrifugeAnalysis(location)

S = SavefilePortal(location);

S.setFrametype(7);

equil = S.nextFrame(); % Load 1st 

tval =  [];
l1val = [];
l2val = [];

for N = 2:S.numFrames();
    f = S.nextFrame();

    tval(end+1) = sum(f.time.history);
    delta = f.mass - equil.mass;
    l1val(end+1) = norm(delta(:),1) / numel(delta);
    l2val(end+1) = norm(delta(:),2) / sqrt(numel(delta));
end

result.T = tval;
result.L1 = l1val;
result.L2 = l2val;


end
