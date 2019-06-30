function util_Frame2NCD(nfile, frame)
% Serializes an Imogen saveframe 'frame' into NetCDF 4 format file 'nfile'

%d3 = @(v) {'nx',size(v,1),'ny',size(v,2),'nz',size(v,3')};
%d3_fixed = {'nx','ny','nz'};
% or nr, ntheta, nz...

ncid = netcdf.create(nfile, '64BIT_OFFSET');

% FIXME: check for r/w failure here

vecdim = netcdf.defDim(ncid, '3elem', 3); % this is dumb but it's historically locked...
scaldim= netcdf.defDim(ncid, '1elem', 1);

% Define time substructure dimensions
tinfo  = netcdf.defDim(ncid, 'tinfo', 5);
tstart = netcdf.defDim(ncid, 'tstart', length(frame.time.started));

% Define parallel substructure dimensions
geomnx = netcdf.defDim(ncid, 'geomnx', size(frame.parallel.geometry,1));
geomny = netcdf.defDim(ncid, 'geomny', size(frame.parallel.geometry,2));
geomnz = netcdf.defDim(ncid, 'geomnz', size(frame.parallel.geometry,3));

% define small parameter dimensions
aboutstr   = netcdf.defDim(ncid, 'aboutstr', length(frame.about));
versionstr = netcdf.defDim(ncid, 'versionstr', length(frame.ver));

% Define dgrid dimensions
dgridnx = netcdf.defDim(ncid, 'dgridnx', size(frame.dGrid{1},1));
dgridny = netcdf.defDim(ncid, 'dgridny', size(frame.dGrid{2},2));
dgridnz = netcdf.defDim(ncid, 'dgridnz', size(frame.dGrid{3},3));
simdim  = netcdf.defDim(ncid, 'simdim', length(frame.dim)); % good lord...

% define grid dimensions
nx = netcdf.defDim(ncid, 'nx', size(frame.mass,1));
ny = netcdf.defDim(ncid, 'ny', size(frame.mass,2));
nz = netcdf.defDim(ncid, 'nz', size(frame.mass,3));

% DEFINE ALL VARIABLES
% Define time info substructure
timeinfo_scals  = netcdf.defVar(ncid, 'timeinfo_scals', 'double', tinfo);
timeinfo_tstart = netcdf.defVar(ncid, 'timeinfo_tstart', 'char', tstart);

% Define parallel substructure
parallel_geom   = netcdf.defVar(ncid, 'parallel_geom', 'double', [geomnx geomny geomnz]);
parallel_gdims  = netcdf.defVar(ncid, 'parallel_gdims', 'double', vecdim);
parallel_offset = netcdf.defVar(ncid, 'parallel_offset', 'double', vecdim);
parallel_halobits=netcdf.defVar(ncid, 'parallel_halobits', 'double', scaldim);
parallel_haloamt= netcdf.defVar(ncid, 'parallel_haloamt', 'double', scaldim);

% Define small parameters
gammavar   = netcdf.defVar(ncid, 'gamma', 'double', scaldim);
aboutvar   = netcdf.defVar(ncid, 'about', 'char', aboutstr);
versionvar = netcdf.defVar(ncid, 'version', 'char', versionstr);

% Define dgrid parameters
dgrid_x = netcdf.defVar(ncid, 'dgrid_x', 'double', dgridnx);
dgrid_y = netcdf.defVar(ncid, 'dgrid_y', 'double', dgridny);
dgrid_z = netcdf.defVar(ncid, 'dgrid_z', 'double', dgridnz);
dimvar  = netcdf.defVar(ncid, 'dim', 'char', simdim); % good grief

% Define main data variables
mass = netcdf.defVar(ncid, 'mass', 'double', [nx ny nz]);
if isfield(frame, 'momX') % saving conservative vars
    momX = netcdf.defVar(ncid, 'momX', 'double', [nx ny nz]);
    momY = netcdf.defVar(ncid, 'momY', 'double', [nx ny nz]);
    momZ = netcdf.defVar(ncid, 'momZ', 'double', [nx ny nz]);
    ener = netcdf.defVar(ncid, 'ener', 'double', [nx ny nz]);
else
    momX = netcdf.defVar(ncid, 'velX', 'double', [nx ny nz]);
    momY = netcdf.defVar(ncid, 'velY', 'double', [nx ny nz]);
    momZ = netcdf.defVar(ncid, 'velZ', 'double', [nx ny nz]);
    ener = netcdf.defVar(ncid, 'eint', 'double', [nx ny nz]);
end

% HACK FIXME - the burning turd around resultsHandler.m:75 is here present.
% HACK FIXME - rather than setup arbitrary-fluid-count handling I just hacked it for 2
if isfield(frame, 'mass2')
    mass2 = netcdf.defVar(ncid, 'mass2', 'double', [nx ny nz]);
    if isfield(frame, 'momX2')
        momX2 = netcdf.defVar(ncid, 'momX2', 'double', [nx ny nz]);
        momY2 = netcdf.defVar(ncid, 'momY2', 'double', [nx ny nz]);
        momZ2 = netcdf.defVar(ncid, 'momZ2', 'double', [nx ny nz]);
        ener2 = netcdf.defVar(ncid, 'ener2', 'double', [nx ny nz]);
    else
        momX2 = netcdf.defVar(ncid, 'velX2', 'double', [nx ny nz]);
        momY2 = netcdf.defVar(ncid, 'velY2', 'double', [nx ny nz]);
        momZ2 = netcdf.defVar(ncid, 'velZ2', 'double', [nx ny nz]);
        ener2 = netcdf.defVar(ncid, 'eint2', 'double', [nx ny nz]);
    end
    
end

magstatus = netcdf.defVar(ncid, 'magstatus', 'double', scaldim);
if isempty(frame.magX) || numel(frame.magX) ~= numel(frame.mass)
    % Defines a placeholder that marks magnetic arrays as absent
else
    magX = netcdf.defVar(ncid, 'magX', 'double', [nx ny nz]);
    magY = netcdf.defVar(ncid, 'magY', 'double', [nx ny nz]);
    magZ = netcdf.defVar(ncid, 'magZ', 'double', [nx ny nz]);
end

netcdf.endDef(ncid);

% Serialize time substructure
netcdf.putVar(ncid, timeinfo_scals, [frame.time.time;frame.time.iterMax;frame.time.timeMax;frame.time.wallMax;frame.time.iteration]);
netcdf.putVar(ncid, timeinfo_tstart, frame.time.started);

% Serialize geometry info
netcdf.putVar(ncid, parallel_geom, frame.parallel.geometry);
netcdf.putVar(ncid, parallel_gdims, frame.parallel.globalDims);
netcdf.putVar(ncid, parallel_offset, frame.parallel.myOffset);
netcdf.putVar(ncid, parallel_halobits, frame.parallel.haloBits);
netcdf.putVar(ncid, parallel_haloamt, frame.parallel.haloAmt);

% Serialize parameters and other small stuff
netcdf.putVar(ncid, gammavar, frame.gamma(1));
netcdf.putVar(ncid, aboutvar, frame.about);
netcdf.putVar(ncid, versionvar, frame.ver);

% Write dGrid information
netcdf.putVar(ncid, dgrid_x, frame.dGrid{1});
netcdf.putVar(ncid, dgrid_y, frame.dGrid{2});
netcdf.putVar(ncid, dgrid_z, frame.dGrid{3});
netcdf.putVar(ncid, dimvar, frame.dim);

netcdf.putVar(ncid, mass, frame.mass);
if isfield(frame, 'momX')
    netcdf.putVar(ncid, momX, frame.momX);
    netcdf.putVar(ncid, momY, frame.momY);
    netcdf.putVar(ncid, momZ, frame.momZ);
    netcdf.putVar(ncid, ener, frame.ener);
else
    netcdf.putVar(ncid, momX, frame.velX);
    netcdf.putVar(ncid, momY, frame.velY);
    netcdf.putVar(ncid, momZ, frame.velZ);
    netcdf.putVar(ncid, ener, frame.eint);
end

if isfield(frame, 'mass2')
    netcdf.putVar(ncid, mass2, frame.mass2);
    if isfield(frame, 'momX2')
        netcdf.putVar(ncid, momX2, frame.momX2);
        netcdf.putVar(ncid, momY2, frame.momY2);
        netcdf.putVar(ncid, momZ2, frame.momZ2);
        netcdf.putVar(ncid, ener2, frame.ener2);
    else
        netcdf.putVar(ncid, momX2, frame.velX2);
        netcdf.putVar(ncid, momY2, frame.velY2);
        netcdf.putVar(ncid, momZ2, frame.velZ2);
        netcdf.putVar(ncid, ener2, frame.eint2);
    end
end

if isempty(frame.magX) || numel(frame.magX) ~= numel(frame.mass)
    netcdf.putVar(ncid, magstatus, 0);
else
    netcdf.putVar(ncid, magstatus, 1);
    netcdf.putVar(ncid, magX, frame.magX);
    netcdf.putVar(ncid, magY, frame.magY);
    netcdf.putVar(ncid, magZ, frame.magZ);
end

netcdf.close(ncid);

end
