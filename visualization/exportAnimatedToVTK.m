function exportAnimatedToVTK(outBasename, inType, range, varset, timeNormalization)
% exportAnimatedToEnsight(outBasename, inBasename, range, varset, timeNormalization)
%>> outBasename: Base filename for output Ensight files, e.g. 'mysimulation'
%>> inBasename:  Input filename for Imogen .mat savefiles, e.g. '2D_XY'
%>> range:       Set of savefiles to export (e.g. 0:50:1000)
%>> varset:      {'names','of','variables'} to save (see util_DerivedQty for list)
%>> timeNormalization: Allows Imogen timestep-time to be converted into characteristic time units
	
addpath('~/vtkwriter');

%--- Interactively fill in missing arguments ---%
if nargin < 5
    fprintf('Not enough input arguments to run automatically. Input them now:\n');
    outBasename = input('Base filename for exported files (e.g. "torus1"): ', 's');
    inType      = input('Frame type to read (e.g. "3D_XYZ", no trailing _):','s');
    range       = input('Range of frames to export; _START = 0 (e.g. 0:50:1000 to do every 50th frame from start to 1000): ');
    varset      = eval(input('Cell array of variable names: ','s'));
    if isempty(varset) || ~isa(varset,'cell'); disp('Not valid; Defaulting to mass, velocity, pressure'); varset={'mass','velocity','pressure'}; end
    timeNormalization = input('Characteristic time to normalize by (e.g. alfven crossing time or characteristic rotation period. If in doubt enter 1): ');
end

SP = SavefilePortal('./', inType);

if range == -1
    fprintf('Defaulting to all frames: %i total\n', int32(SP.numFrames()));
    range = 1:SP.numFrames();
end

pertonly = 0;%input('Export perturbed quantities (1) or full (0)? ');

%--- Initialization ---%
fprintf('Beginning export of %i files\n', numel(range));
%exportedFrameNumber = 0;

equilframe = [];

% Runs in parallel if MPI has been started
if mpi_isinitialized() == 0
  mpi_init();
end

minf = mpi_basicinfo();
nworkers = minf(1); myworker = minf(2);

ntotal = numel(range); % number of frames to write
nstep = nworkers;

if nworkers > 1
    fprintf('Work distributed among %i workers.\n', nworkers);
end

tic;

stepnums = zeros([ntotal 1]);

% GET GEOMETRY DATA HERE
d = SP.getInitialConditions();
g = GeometryManager(d.ini.geometry.globalDomainRez);
g.geometryCylindrical(d.ini.geometry.affine(1), round(2*pi/(d.ini.geometry.d3h(2)*d.ini.geometry.globalDomainRez(2))), d.ini.geometry.d3h(1), d.ini.geometry.affine(2), d.ini.geometry.d3h(3));
[xp, yp, zp] = g.ndgridSetIJK('pos','square'); %#ok<ASGLU>

% WRITE GEOMETRY STRING:
% vtkwrite(filename, 'structured_grid', geomX, geomY, geomZ
outcmdstr = sprintf('vtkwrite(''%s%%04i.vtk'', ''structured_grid'', xp, yp, zp', outBasename);

ITER = myworker+1;
dataframe = SP.setFrame(ITER); 

framedat = {};
for vn = 1:numel(varset)
        framedat{vn} = util_DerivedQty(dataframe, varset{vn}, 0);
end

% Loop over output varset:
% append to the vtkwrite string
for vn = 1:numel(varset)
    if isfield(framedat{vn}, 'X') % if isVector...
        outcmdstr = sprintf('%s, ''vectors'', ''%s'', framedat{%i}.X, framedat{%i}.Y, framedat{%i}.Z', outcmdstr, varset{vn}, vn, vn, vn);
    else
        outcmdstr = sprintf('%s, ''scalars'', ''%s'', framedat{%i}', outcmdstr, varset{vn}, vn);
    end
end
outcmdstr = sprintf('%s, ''binary'');', outcmdstr);

%framedat = {};

%--- Loop over all frames ---%
for ITER = (myworker+1):nstep:ntotal
    if ITER ~= (myworker+1); dataframe = SP.setFrame(ITER); end

% FIXME this fails in parallel horribly...
    stepnums(ITER) = sum(dataframe.time.history) / timeNormalization;

    finalstr = sprintf(outcmdstr, ITER);
    % for elements of varset,
    % fetch using util_DerivedQty into a cell array
    if ITER ~= (myworker +1)
        for vn = 1:numel(varset)
            framedat{vn} = util_DerivedQty(dataframe, varset{vn}, 0); %#ok<AGROW>
        end
    end
    
    eval(finalstr);
    
    fprintf('%i ',myworker);
end

fprintf('Rank %i finished in %g sec.\n', myworker, toc);

end

function out = subtractEquil(in, eq)
out = in;

out.mass = in.mass - eq.mass;
out.ener = in.ener - eq.ener;

out.momX = in.momX - eq.momX;
out.momY = in.momY - eq.momY;
out.momZ = in.momZ - eq.momZ;

out.magX = in.magX - eq.magX;
out.magY = in.magY - eq.magY;
out.magZ = in.magZ - eq.magZ;

end
