function util_Frame2HDF(hname, frame)
% function util_Frame2HDF(hname, frame)
% saves the Imogen data frame 'frame' to the file named by 'hname' as an HDF5 file

% Save the 'time' struct
h5create(hname, '/dgridx', size(frame.dGrid{1}), 'Datatype', 'double');
h5create(hname, '/dgridy', size(frame.dGrid{2}), 'Datatype', 'double');
h5create(hname, '/dgridz', size(frame.dGrid{3}), 'Datatype', 'double');

if isfield(frame, 'momX') || isprop(frame, 'momX')
    fluvars = {'mass', 'momX', 'momY', 'momZ', 'ener'};
else
    fluvars = {'mass', 'velX', 'velY', 'velZ', 'eint'};
end
for i = 1:5
    h5create(hname, ['/fluid1/' fluvars{i}], size(frame.(fluvars{i})), 'Datatype', 'double');
end

% Make a placeholder for time attributes
h5create(hname, '/timehist', 1, 'Datatype','double');

if isfield(frame, 'mass2')
    if isfield(frame, 'momX2') || isprop(frame, 'momX2')
        flu2vars = {'mass2', 'momX2', 'momY2', 'momZ2', 'ener2'};
    else
        flu2vars = {'mass2', 'velX2', 'velY2', 'velZ2', 'eint2'};
    end
    for i = 1:5
        h5create(hname, ['/fluid2/' fluvars{i}], size(frame.(flu2vars{i})), 'Datatype', 'double');
    end
end

if numel(frame.magX)==0; frame.magX = 0; frame.magY = 0; frame.magZ = 0; end

h5create(hname, '/mag/X', size(frame.magX));
h5create(hname, '/mag/Y', size(frame.magY));
h5create(hname, '/mag/Z', size(frame.magZ));

h5write(hname, '/timehist', [1]);

timeatts = {'time', 'iterMax', 'timeMax', 'wallMax', 'iteration', 'started'};
for i = 1:5
    h5writeatt(hname, '/timehist', timeatts{i}, frame.time.(timeatts{i}));
end

paratts = {'geometry', 'globalDims', 'myOffset', 'haloBits', 'haloAmt'};

for i = 1:5
    h5writeatt(hname, '/', ['par_ ' paratts{i}], frame.parallel.(paratts{i}));
end

h5writeatt(hname, '/', 'gamma', frame.gamma);

h5writeatt(hname, '/', 'about', frame.about(:)');
h5writeatt(hname, '/', 'ver', frame.ver(:)');

% handle this in a reasonably graceful manner... not that we'll ever get nonscalars for these again. HAH!
h5write(hname, '/dgridx', frame.dGrid{1});
h5write(hname, '/dgridy', frame.dGrid{2});
h5write(hname, '/dgridz', frame.dGrid{3});

% is this even relevant? lol!
%frame.dim

% dump fluid 1, it always exists...
for i = 1:5
    h5write(hname, ['/fluid1/' fluvars{i}], frame.(fluvars{i}));
end

if isfield(frame, 'mass2')
    for i = 1:5
        h5write(hname, ['/fluid2/' fluvars{i}], frame.(flu2vars{i}));
    end
end

h5write(hname, '/mag/X', frame.magX);
h5write(hname, '/mag/Y', frame.magY);
h5write(hname, '/mag/Z', frame.magZ);

end
