function frame = util_HDF2Frame(hname)
%function frame = util_Frame2HDF(hname)
% loads the hdf5 file at 'hname' and returns Imogen saveframe 'frame'.

frame.time = struct('history',[], 'time', [], 'iterMax', [], 'timeMax', [], 'wallMax', [], 'iteration', [], 'started', []);
frame.time.history = h5read(hname, '/timehist');

timeatts = {'time', 'iterMax', 'timeMax', 'wallMax', 'iteration', 'started'};
for i = 1:5
    frame.time.(timeatts{i}) = h5readatt(hname, '/timehist', timeatts{i});
end

% this shouldn't be here, but it was before so it still is.
frame.iter = h5readatt(hname, '/', 'iter');

% Grab the parallel struct info.
% This is dumped into the global directory as attributes because these are all small scalars... okay whatever.
frame.parallel = struct('geometry', [], 'globalDims', [], 'myOffset', [], 'haloBits', [], 'haloAmt', []);
paratts = {'geometry', 'globalDims', 'myOffset', 'haloBits', 'haloAmt'};

for i = 1:5
    frame.parallel.(paratts{i}) = h5readatt(hname, '/', ['par_ ' paratts{i}]);
end
frame.parallel.globalDims = transpose(frame.parallel.globalDims);
frame.parallel.myOffset = transpose(frame.parallel.myOffset);


frame.gamma = h5readatt(hname, '/', 'gamma');

frame.about = h5readatt(hname, '/', 'about');
frame.ver   = h5readatt(hname, '/', 'ver');

frame.dGrid = cell([1 3]);
frame.dGrid{1} = h5read(hname, '/dgridx');
frame.dGrid{2} = h5read(hname, '/dgridy');
frame.dGrid{3} = h5read(hname, '/dgridz');


frame.mass = h5read(hname, '/fluid1/mass');
frame.momX = h5read(hname, '/fluid1/momX');
frame.momY = h5read(hname, '/fluid1/momY');
frame.momZ = h5read(hname, '/fluid1/momZ');
frame.ener = h5read(hname, '/fluid1/ener');

q = h5info(hname, '/');

if ish5group(q, '/fluid2')
    frame.mass2 = h5read(hname, '/fluid2/mass');
    frame.momX2 = h5read(hname, '/fluid2/momX');
    frame.momY2 = h5read(hname, '/fluid2/momY');
    frame.momZ2 = h5read(hname, '/fluid2/momZ');
    frame.ener2 = h5read(hname, '/fluid2/ener');
end

frame.magX = h5read(hname, '/mag/X');
frame.magY = h5read(hname, '/mag/Y');
frame.magZ = h5read(hname, '/mag/Z');

end

function tf = ish5group(info, name)

for q = 1:numel(info.Groups)
    if strcmp(info.Groups(q).Name, name); tf = 1; return; end
end

tf = 0;

end
