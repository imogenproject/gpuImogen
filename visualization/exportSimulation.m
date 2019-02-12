function exportSimulation(outBasename, inType, range, varset, timeNormalization, reverseIndexOrder)
% exportAnimatedToEnsight(outBasename, inBasename, range, varset, timeNormalization)
%>> outBasename: Base filename for output Ensight files, e.g. 'mysimulation'
%>> inBasename:  Input filename for Imogen .mat savefiles, e.g. '2D_XY'
%>> range:       Set of savefiles to export (e.g. 0:50:1000)
%>> varset:      {'names','of','variables'} to save (see util_DerivedQty for list)
%>> timeNormalization: Allows Imogen timestep-time to be converted into characteristic time units

%--- Interactively fill in missing arguments ---%
if nargin < 5
    fprintf('Not enough input arguments to run automatically. Input them now:\n');
    outBasename = input('Base filename for exported files (e.g. "torus1"): ', 's');
    inType      = input('Frame type to read (e.g. "3D_XYZ", no trailing _):','s');
    range       = input('Range of frames to export (e.g. 0:50:1000 to do every 50th frame from start to 1000): ');
    varset      = eval(input('Cell array of variable names to convert: ','s'));
    if isempty(varset) || ~isa(varset,'cell'); disp('Not valid; Defaulting to mass, velocity, pressure'); varset={'mass','velocity','pressure'}; end
    timeNormalization = input('Characteristic time to normalize by (e.g. alfven crossing time or characteristic rotation period. If in doubt enter 1): ');
    reverseIndexOrder = input('1 to rearrange [XYZ] to [ZYX], 0 to not: ');
end

if reverseIndexOrder
    warning('WARNING: reverseIndexOrder requested but exportAnimatedToVTK does not support this. All output will retain normal XYZ X-linear-stride order.');
end

SP = SavefilePortal('./', inType);

if range == -1
    fprintf('Defaulting to all frames: %i total\n', int32(SP.numFrames));
    range = 1:SP.numFrames;
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

if 0
    exportAnimatedToEnsight(SP, outBasename, range, varset, timeNormalization, reverseIndexOrder);
end
if 0
    exportAnimatedToVTK(SP, outBasename, range, varset, timeNormalization, reverseIndexOrder);
end
if 1
    exportAnimatedToXDMF(SP, outBasename, range, varset, timeNormalization, reverseIndexOrder);
end
    
end

