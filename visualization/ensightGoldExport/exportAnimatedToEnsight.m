function exportAnimatedToEnsight(outBasename, inBasename, padlength, range, timeNormalization, autoparallel)
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

%range = removeNonexistantEntries(inBasename, padlength, range);
maxFrameno = max(range);

equilframe = [];

% If running under automatic or not sure, ask about doing this in parallel
% Note that trying this if MPI isn't started equals CRASH COREDUMP TIME
if (nargin < 6) || (autoparallel == 0)
  autoparallel = input('Attempt to run in parallel (1/0)? ');
end
if autoparallel == 1; minf = mpi_basicinfo(); else; minf = [1 0]; end

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
    % Take first guess; Always replace _START
    fname = sprintf('%s_%0*i', inBasename, padlength, range(ITER));
    fprintf('Rank %i exporting %s as frame %i: %g; ', minf(2), fname, ITER, toc);

    dataframe = util_LoadWholeFrame(inBasename, padlength, range(ITER));

 %   if (ITER == 1) && (pertonly == 1)
 %       equilframe = dataframe;
 %   end

%    if pertonly == 1
%        dataframe = subtractEquil(dataframe, equilframe);
%    end
    fprintf('Rank %i load: %g; ', minf(2), toc);
    writeEnsightDatafiles(outBasename, ITER-1, dataframe);
    if range(ITER) == maxFrameno
        writeEnsightMasterFiles(outBasename, range, dataframe, timeNormalization);
    end

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


function newrange = removeNonexistantEntries(inBasename, padlength, range)

existrange = [];

for ITER = 1:numel(range)
    % Take first guess; Always replace _START
    fname = sprintf('%s_rank0_%0*i.mat', inBasename, padlength, range(ITER));
    if range(ITER) == 0; fname = sprintf('%s_rank0_START.mat', inBasename); end

    % Check existance; if fails, try _FINAL then give up
    doesExist = exist(fname, 'file');
    if (doesExist == 0) & (range(ITER) == max(range))
        fname = sprintf('%s_rank0_FINAL.mat', inBasename);
        doesExist = exist(fname, 'file');
    end
    
    if doesExist ~= 0; existrange(end+1) = ITER; end
end

newrange = range(existrange);
if numel(newrange) ~= numel(range);
    fprintf('WARNING: Removed %i entries that could not be opened from list.\n', numel(range)-numel(newrange));
end

if numel(newrange) == 0;
   error('UNRECOVERABLE: No files indicated existed. Perhaps remove trailing _ from base name?\n'); 
end

end
