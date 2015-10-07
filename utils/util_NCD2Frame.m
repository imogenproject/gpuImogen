function frame = util_NCD2Frame(nfile)
% Serializes an Imogen saveframe 'frame' into NetCDF 4 format file 'nfile'

% Deserialize time substructure
frame.time.history = ncread(nfile,'timeinfo_hist');
  ts = ncread(nfile,'timeinfo_scals'); % time substruct scalars
  frame.time.time = ts(1);
  frame.time.iterMax = ts(2);
  frame.time.timeMax = ts(3);
  frame.time.wallMax = ts(4);
  frame.time.iteration = ts(5);
  frame.iter = frame.time.iteration;

frame.time.started = ncread(nfile,'timeinfo_tstart')';

% Deserialize parallel substructure

frame.parallel.geometry   = ncread(nfile, 'parallel_geom');
frame.parallel.globalDims = ncread(nfile, 'parallel_gdims')';
frame.parallel.myOffset   = ncread(nfile, 'parallel_offset')';

% Deserialize small stuff

frame.gamma   = ncread(nfile, 'gamma');
frame.about   = ncread(nfile, 'about')';
frame.version = ncread(nfile, 'version')';

% Note: copy frame.time.iteration above back to frame.iter upon load

frame.dGrid{1} = ncread(nfile, 'dgrid_x');
frame.dGrid{2} = ncread(nfile, 'dgrid_y');
frame.dGrid{3} = ncread(nfile, 'dgrid_z');

frame.dim = ncread(nfile, 'dim');

% The main event: Deserialize the data arrays.

frame.mass = ncread(nfile,'mass');
frame.momX = ncread(nfile,'momX');
frame.momY = ncread(nfile,'momY');
frame.momZ = ncread(nfile,'momZ');
frame.ener = ncread(nfile,'ener');

magstat = ncread(nfile,'magstatus');

if magstat == 0
    frame.magX = 0; frame.magY = 0; frame.magZ = 0;
else
    frame.magX = ncread(nfile,'magX');
    frame.magY = ncread(nfile,'magY');
    frame.magZ = ncread(nfile,'magZ');
end

end

