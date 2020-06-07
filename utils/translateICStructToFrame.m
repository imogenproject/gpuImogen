function F = translateICStructToFrame(IC)

% Create the frame structure and copy the main fluid state arrays
if numel(IC.fluids) == 1
    F = struct('time',[],'parallel',[],'gamma',[],'about',[],'ver',[],'dGrid',[],'mass',[],'momX',[],'momY',[],'momZ',[]);

    a = {'mass','momX','momY','momZ','ener'};
    for i = 1:5; F.(a{i}) = IC.fluids(1).(a{i}); end
    
else
    F = struct('time',[],'parallel',[],'gamma',[],'about',[],'ver',[],'dGrid',[],'mass',[],'momX',[],'momY',[],'momZ',[],'mass2',[],'ener2',[],'momX2',[],'momY2',[],'momZ2');

    a = {'mass','momX','momY','momZ','ener'};
    for i = 1:5; F.(a{i}) = IC.fluids(1).(a{i}); end

    b = {'mass2','momX2','momY2','momZ2','ener2'};
    for i = 1:5; F.(b{i}) = IC.fluids(2).(a{i}); end

end

if all([IC.magX(:); IC.magY(:); IC.magZ(:)] == 0)
    % B is zero
    F.magX = 0; F.magY = 0; F.magZ = 0;
else
    F.magX = IC.magX;
    F.magY = IC.magY;
    F.magZ = IC.magZ;
end

h = IC.ini.geometry.d3h;
F.dGrid = cell(3,1);
F.dGrid{1} = h(1);
F.dGrid{2} = h(2);
F.dGrid{3} = h(3);

F.time = struct('time',0,'iterMax',IC.ini.iterMax,'timeMax', IC.ini.timeMax, 'wallMax', IC.ini.wallMax, 'iteration', 0, 'started', []);
F.gamma = IC.fluids(1).details.gamma;
F.about = IC.ini.info;
F.ver = '1.0.1';

g = GPUManager.getInstance();

F.parallel = struct('geometry', IC.ini.geometry.pGeometryType, 'globalDims', IC.ini.geometry.globalDomainRez, 'myOffset', IC.ini.geometry.pLocalDomainOffset, 'haloBits', 0, 'haloAmt', g.useHalo);

end

