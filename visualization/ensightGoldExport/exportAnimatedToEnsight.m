function exportAnimatedToEnsight(outBasename, inType, range, varset, timeNormalization)
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
    range       = input('Range of frames to export; _START = 0 (e.g. 0:50:1000 to do every 50th frame from start to 1000): ');
    varset      = eval(input('Cell array of variable names: ','s'));
    if isempty(varset) || ~isa(varset,'cell'); disp('Not valid; Defaulting to mass, velocity, pressure'); varset={'mass','velocity','pressure'}; end
    timeNormalization = input('Characteristic time to normalize by (e.g. alfven crossing time or characteristic rotation period. If in doubt enter 1): ');
end

SP = SavefilePortal('./', inType);

if range == -1;
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

fprintf('Work distributed among %i workers.\n', nworkers);

tic;

stepnums = zeros([ntotal 1]);

%--- Loop over all frames ---%
for ITER = (myworker+1):nstep:ntotal
    dataframe = SP.setFrame(ITER); 

% FIXME this fails in parallel horribly...
    stepnums(ITER) = sum(dataframe.time.history);

    writeEnsightDatafiles(outBasename, ITER-1, dataframe, varset);
    if ITER == ntotal
        writeEnsightMasterFiles(outBasename, range, SP, varset, timeNormalization);
    end
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

