function util_Frame2NCD(frame, nfile)
% Serializes an Imogen saveframe 'frame' into NetCDF 4 format file 'nfile'

d3 = @(v) {'nx',size(v,1),'ny',size(v,2),'nz',size(v,3')};

d3_fixed = {'nx','ny','nz'};

th = frame.time.history;
if isempty(frame.time.history); th = 0; end

% Serialize time substructure
nccreate(nfile,'timeinfo_hist','Dimensions',{'dthist',numel(th)});
ncwrite(nfile,'timeinfo_hist',th);

nccreate(nfile,'timeinfo_scals','Dimensions',{'tinfo',5});
ncwrite(nfile,'timeinfo_scals',[frame.time.time;frame.time.iterMax;frame.time.timeMax;frame.time.wallMax;frame.time.iteration]);

nccreate(nfile,'timeinfo_tstart','Datatype','char','Dimensions',{'tstart',length(frame.time.started)});
ncwrite(nfile,'timeinfo_tstart',frame.time.started);

% Serialize parallel substructure
dgeo = {'geomnx', size(frame.parallel.geometry,1), 'geomny', size(frame.parallel.geometry,2), 'geomnz', size(frame.parallel.geometry,3)};

nccreate(nfile,'parallel_geom','Dimensions',dgeo );
ncwrite(nfile,'parallel_geom',frame.parallel.geometry);

nccreate(nfile,'parallel_gdims','Dimensions',{'3elem',3});
ncwrite(nfile,'parallel_gdims',frame.parallel.globalDims);

nccreate(nfile,'parallel_offset','Dimensions',{'3elem'});
ncwrite(nfile,'parallel_offset',frame.parallel.myOffset);

% Serialize small stuff

nccreate(nfile,'gamma');
ncwrite(nfile,'gamma',frame.gamma);

nccreate(nfile,'about','Datatype','char','Dimensions',{'aboutstr',length(frame.about)});
ncwrite(nfile,'about',frame.about);

nccreate(nfile,'version','Datatype','char','Dimensions',{'versionstr', length(frame.ver)});
ncwrite(nfile,'version',frame.ver);

% Note: copy frame.time.iteration above back to frame.iter upon load

dgs = {'dgridx',size(frame.dGrid{1},1),'dgridy',size(frame.dGrid{1},2),'dgridz',size(frame.dGrid{1},3)};
dgs0 = {'dgridx','dgridy','dgridz'};

nccreate(nfile,'dgrid_x','Dimensions',dgs);
ncwrite(nfile,'dgrid_x',frame.dGrid{1});

nccreate(nfile,'dgrid_y','Dimensions',dgs0);
ncwrite(nfile,'dgrid_y',frame.dGrid{2});

nccreate(nfile,'dgrid_z','Dimensions',dgs0);
ncwrite(nfile,'dgrid_z',frame.dGrid{3});

nccreate(nfile,'dim','Datatype','char','Dimensions',{'dimsx',3});
ncwrite(nfile,'dim',frame.dim);

% The main event: Serialize the data arrays.

simdim = d3(frame.mass);

nccreate(nfile,'mass','Dimensions',simdim);
ncwrite(nfile,'mass', frame.mass);

nccreate(nfile,'momX','Dimensions',d3_fixed);
ncwrite(nfile,'momX', frame.momX);

nccreate(nfile,'momY','Dimensions',d3_fixed);
ncwrite(nfile,'momY', frame.momY);

nccreate(nfile,'momZ','Dimensions',d3_fixed);
ncwrite(nfile,'momZ', frame.momZ);

nccreate(nfile,'ener','Dimensions',d3_fixed);
ncwrite(nfile,'ener', frame.ener);

if isempty(frame.magX)
    nccreate(nfile,'magstatus');
    ncwrite(nfile,'magstatus',0);
else
    nccreate(nfile,'magstatus');
    ncwrite(nfile,'magstatus',1);

    nccreate(nfile,'magX','Dimensions',d3_fixed);
    ncwrite(nfile,'magX', frame.magX);

    nccreate(nfile,'magY','Dimensions',d3_fixed);
    ncwrite(nfile,'magY', frame.magY);

    nccreate(nfile,'magZ','Dimensions',d3_fixed);
    ncwrite(nfile,'magZ', frame.magZ);
end


end

function d = dims3d(v)

d = {'x',size(v,1),'y',size(v,2),'z',size(v,3')};

end
