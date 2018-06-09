function appendAnimatedToEnsight(outBasename, inBasename, oldrange, addrange, timeNormalization)
%>> outBasename:       Base filename for output Ensight files
%>> inBasename:        Input filename for Imogen .mat savefiles
%>> range:             Set of .mats to export
%>> timeNormalization: Allows Imogen timestep-time to be converted into characteristic time units
	
%--- Initialization ---%
fprintf('Beginning export of %i files\n', numel(addrange));
%exportedFrameNumber = numel(oldrange);

if max(round(addrange) - addrange) ~= 0; error('ERROR: Frame range is not integer-valued.\n'); end
if min(addrange) < 0; error('ERROR: Frame range must be nonnegative.\n'); end

% Automatically remove tailing _ but scorn the user
if inBasename(end) == '_'
    inBasename = inBasename(1:(end-1));
    warning('inBasename had a trailing _; This causes a no-frames-found error & has been automatically stripped out.');
end

frmexists = util_checkFrameExistence(inBasename, addrange);
if all(~frmexists)
  fprintf('No frames to be added exist; Returning.\n');
  return
end

addrange = addrange(frmexists == 1);
maxFrameno = max(addrange);

if nargin == 4; timeNormalization = 1; end

%--- Loop over given frame range ---%
parfor ITER = 1:numel(addrange)
    fprintf('Exporting %s as frame %i... ', fname,  numel(oldrange)+ITER+1);

    dataframe = util_LoadWholeFrame(inBasename, addrange(ITER));

    writeEnsightDatafiles(outBasename, numel(oldrange)+ITER+1, dataframe);
    if addrange(ITER) == maxFrameno
        writeEnsightMasterFiles(outBasename, [oldrange, addrange], dataframe, timeNormalization);
    end

    fprintf('done.\n');
end

end
