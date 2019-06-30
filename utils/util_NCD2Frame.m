function frame = util_NCD2Frame(nfile, options)
% Deserializes an Imogen NC4 file into a saveframe

if nargin < 2; options = 'nothing'; end

metaonly = strcmpi(options, 'metaonly');

ncid = netcdf.open(nfile,'NC_NOWRITE');

% Deserialize time substructure
v = netcdf.inqVarID(ncid, 'timeinfo_scals');
ts = netcdf.getVar(ncid, v);
  frame.time.time = ts(1);
  frame.time.iterMax = ts(2);
  frame.time.timeMax = ts(3);
  frame.time.wallMax = ts(4);
  frame.time.iteration = ts(5);
  frame.iter = frame.time.iteration;

v = netcdf.inqVarID(ncid, 'timeinfo_tstart');
frame.time.started = netcdf.getVar(ncid, v)';

% Deserialize parallel substructure
v = netcdf.inqVarID(ncid, 'parallel_geom');
frame.parallel.geometry = netcdf.getVar(ncid, v);
v = netcdf.inqVarID(ncid, 'parallel_gdims');
frame.parallel.globalDims = netcdf.getVar(ncid, v)';
v = netcdf.inqVarID(ncid, 'parallel_offset');
frame.parallel.myOffset = netcdf.getVar(ncid, v)';
v = netcdf.inqVarID(ncid, 'parallel_halobits');
frame.parallel.haloBits = netcdf.getVar(ncid, v);
v = netcdf.inqVarID(ncid, 'parallel_haloamt');
frame.parallel.haloAmt = netcdf.getVar(ncid, v);

% Deserialize small stuff
v = netcdf.inqVarID(ncid, 'gamma');
frame.gamma = netcdf.getVar(ncid, v);
v = netcdf.inqVarID(ncid, 'about');
frame.about = netcdf.getVar(ncid, v)';
v = netcdf.inqVarID(ncid, 'version');
frame.ver = netcdf.getVar(ncid, v)';

% Note: copy frame.time.iteration above back to frame.iter upon load
v = netcdf.inqVarID(ncid, 'dgrid_x');
frame.dGrid{1} = netcdf.getVar(ncid, v);
v = netcdf.inqVarID(ncid, 'dgrid_y');
frame.dGrid{2} = netcdf.getVar(ncid, v);
v = netcdf.inqVarID(ncid, 'dgrid_z');
frame.dGrid{3} = netcdf.getVar(ncid, v);

v = netcdf.inqVarID(ncid, 'dim');
frame.dim = netcdf.getVar(ncid, v);

if metaonly
    % Injects potentially needful information which otherwise requires the vars to be present
    try
        vf = netcdf.inqVarId(ncid, 'momX');
    catch
        vf = -1234;
    end
    if vf ~= -1234; frame.varFmt = 'conservative'; else; frame.varFmt = 'primitive'; end
    
    try
        vf = netcdf.inqVarID(ncid, 'mass2');
    catch
        vf = -1234;
    end
    if vf ~= -1234; frame.twoFluids = 1; else; frame.twoFluids = 0; end
    
    netcdf.close(ncid);
    return;
end

% The main event: Deserialize the data arrays.
v = netcdf.inqVarID(ncid, 'mass');
frame.mass = netcdf.getVar(ncid, v);
try
    vf = netcdf.inqVarID(ncid, 'momX');
catch
    vf = -1234;
end

if vf ~= -1234
    v = netcdf.inqVarID(ncid, 'momX');
    frame.momX = netcdf.getVar(ncid, v);
    v = netcdf.inqVarID(ncid, 'momY');
    frame.momY = netcdf.getVar(ncid, v);
    v = netcdf.inqVarID(ncid, 'momZ');
    frame.momZ = netcdf.getVar(ncid, v);
    v = netcdf.inqVarID(ncid, 'ener');
    frame.ener = netcdf.getVar(ncid, v);
else
    v = netcdf.inqVarID(ncid, 'velX');
    frame.velX = netcdf.getVar(ncid, v);
    v = netcdf.inqVarID(ncid, 'velY');
    frame.velY = netcdf.getVar(ncid, v);
    v = netcdf.inqVarID(ncid, 'velZ');
    frame.velZ = netcdf.getVar(ncid, v);
    v = netcdf.inqVarID(ncid, 'eint');
    frame.eint = netcdf.getVar(ncid, v);
end

% netcdf-matlab is... not very smart.
try
    v = netcdf.inqVarID(ncid, 'mass2');
catch
    v = -1234;
end

if v ~= -1234
    v = netcdf.inqVarID(ncid, 'mass2');
    frame.mass2 = netcdf.getVar(ncid, v);
    if vf ~= -1234
        v = netcdf.inqVarID(ncid, 'momX2');
        frame.momX2 = netcdf.getVar(ncid, v);
        v = netcdf.inqVarID(ncid, 'momY2');
        frame.momY2 = netcdf.getVar(ncid, v);
        v = netcdf.inqVarID(ncid, 'momZ2');
        frame.momZ2 = netcdf.getVar(ncid, v);
        v = netcdf.inqVarID(ncid, 'ener2');
        frame.ener2 = netcdf.getVar(ncid, v);
    else
        v = netcdf.inqVarID(ncid, 'velX2');
        frame.velX2 = netcdf.getVar(ncid, v);
        v = netcdf.inqVarID(ncid, 'velY2');
        frame.velY2 = netcdf.getVar(ncid, v);
        v = netcdf.inqVarID(ncid, 'velZ2');
        frame.velZ2 = netcdf.getVar(ncid, v);
        v = netcdf.inqVarID(ncid, 'eint2');
        frame.eint2 = netcdf.getVar(ncid, v);
    end
end

try
    v = netcdf.inqVarID(ncid, 'magstatus');
    magstat = ncread(nfile,'magstatus');
catch NOTTHERE %#ok<NASGU>
    magstat = 0;
end

if magstat == 0
    frame.magX = 0; frame.magY = 0; frame.magZ = 0;
else
    v = netcdf.inqVarID(ncid, 'magX');
    frame.magX = netcdf.getVar(ncid, v);
    v = netcdf.inqVarID(ncid, 'magY');
    frame.magY = netcdf.getVar(ncid, v);
    v = netcdf.inqVarID(ncid, 'magZ');
    frame.magY = netcdf.getVar(ncid, v);
end

netcdf.close(ncid);

end

