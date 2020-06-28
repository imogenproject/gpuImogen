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

nProcs = IC.ini.geomgr.topology.nproc;
coord = IC.ini.geomgr.topology.coord;
procgrid = zeros(nProcs);
np = prod(nProcs);
procgrid(1:np) = 0:(np-1);

halobits = 0;
B = BCManager();
bs = B.expandBCStruct(IC.ini.bcMode{1});

q = @B.bcModeToNumber;
bc1 = [q(bs{1,1}) q(bs{2,1})  q(bs{1,2}) q(bs{2,2})   q(bs{1,3}) q(bs{2,3})];
npm = nProcs - 1;

if (nProcs(1) > 1) && ( (bc1(1) == 1) || (coord(1) > 0) ); halobits = halobits + 1; end
if (nProcs(1) > 1) && ( (bc1(2) == 1) || (coord(1) < npm(1)) ); halobits = halobits + 2; end
if (nProcs(2) > 1) && ( (bc1(3) == 1) || (coord(2) > 0) ); halobits = halobits + 4; end
if (nProcs(2) > 1) && ( (bc1(4) == 1) || (coord(2) < npm(2)) ); halobits = halobits + 8; end
if (nProcs(3) > 1) && ( (bc1(5) == 1) || (coord(3) > 0) ); halobits = halobits + 16; end
if (nProcs(3) > 1) && ( (bc1(6) == 1) || (coord(3) < npm(3)) ); halobits = halobits + 32; end

F.parallel = struct('geometry', procgrid, 'globalDims', IC.ini.geometry.globalDomainRez, 'myOffset', IC.ini.geometry.pLocalDomainOffset, 'haloBits', int64(halobits), 'haloAmt', g.useHalo);

end

