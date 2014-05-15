function exportAnimatedToEnsight(outBasename, inBasename, padlength, range, timeNormalization)
%>> outBasename:       Base filename for output Ensight files
%>> inBasename:        Input filename for Imogen .mat savefiles
%>> padlength:         Number of zeros in Imogen filenames
%>> range:             Set of .mats to export
%>> timeNormalization: Allows Imogen timestep-time to be converted into characteristic time units
	
%--- Interactively fill in missing arguments ---%
if nargin < 4
    fprintf('Not enough input arguments to run automatically.\n');
    outBasename = input('Base filename for exported files (e.g. "torus1"): ', 's');
    inBasename  = input('Base filename for source files, (e.g. "3D_XYZ", no trailing _):','s');
    padlength   = input('Length of frame #s in source files (e.g. 3D_XYZ_xxxx -> 4): ');
    range       = input('Range of frames to export; _START = 0 (e.g. 0:50:1000 to do every 50th frame from start to 1000): ');
    timeNormalization = input('Characteristic time to normalize by (e.g. alfven crossing time or characteristic rotation period. If in doubt enter 1): ');
end

pertonly = 0;%input('Export perturbed quantities (1) or full (0)? ');

%--- Initialization ---%
fprintf('Beginning export of %i files\n', numel(range));
exportedFrameNumber = 0;

if max(round(range) - range) ~= 0; error('ERROR: Frame range is not integer-valued.\n'); end
if min(range) < 0; error('ERROR: Frame range must be nonnegative.\n'); end

frmexists = util_checkFrameExistence(inBasename, padlength, range);

fprintf('Found %i/%i frames to exist.\n',numel(find(frmexists)),numel(frmexists));

if all(~frmexists);
    fprintf('No frames match patten; Aborting.\n');
    return
end

range      = range(frmexists == 1);
maxFrameno = max(range);
equilframe = [];

% Runs in parallel if MPI has been started
if mpi_isinitialized() == 0
  minf = [1 0];
else
  minf = mpi_basicinfo();
end

% Attempt to most evenly partition work among workers
ntotal = numel(range); % number of frames to write
nforme = floor(ntotal/minf(1)); % minimum each rank must take
nadd = ntotal - nforme * minf(1); % number left over

fprintf('Work distributed among %i workers.\n', minf(1));

if minf(2) < nadd % I must take one more
  nforme = nforme + 1;
  ninit = minf(2)*nforme;

  localrange = (ninit+1):(ninit+nforme);
else
  ninit = nadd + minf(2)*nforme;
  localrange = (ninit+1):(ninit+nforme);
end

tic;
fprintf('Rank %i exporting frames %i to %i inclusive.\n', minf(2),localrange(1),localrange(end));

%--- Loop over given frame range ---%
for ITER = ninit+(1:nforme)
    dataframe = util_LoadWholeFrame(inBasename, padlength, range(ITER));

 %   if (ITER == 1) && (pertonly == 1)
 %       equilframe = dataframe;
 %   end

%    if pertonly == 1
%        dataframe = subtractEquil(dataframe, equilframe);
%    end
    writeEnsightDatafiles(outBasename, ITER-1, dataframe);
    if range(ITER) == maxFrameno
        writeEnsightMasterFiles(outBasename, range, dataframe, timeNormalization);
    end
    fprintf('%i',minf(2));

end

fprintf('Rank %i finished. total %g.\n', minf(2), toc);

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

