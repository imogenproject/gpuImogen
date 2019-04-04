function exportAnimatedToEnsight(SP, outBasename, range, varset, timeNormalization, reverseIndexOrder)
% exportAnimatedToEnsight(SP, outBasename, inBasename, range, varset, timeNormalization)
%>> SP: SavefilePortal input
%>> outBasename: Base filename for output Ensight files, e.g. 'mysimulation'
%>> inBasename:  Input filename for Imogen .mat savefiles, e.g. '2D_XY'
%>> range:       Set of savefiles to export (e.g. 0:50:1000)
%>> varset:      {'names','of','variables'} to save
%>> timeNormalization: Allows Imogen timestep-time to be converted into characteristic time units
	
%--- Interactively fill in missing arguments ---%
if nargin < 6
    error('Access this using exportSimulation');
end

pertonly = 0;%input('Export perturbed quantities (1) or full (0)? ');

equilframe = [];

minf = mpi_basicinfo();
nworkers = minf(1); myworker = minf(2);
ntotal = numel(range); % number of frames to write
nstep = nworkers;

tic;

stepnums = zeros([ntotal 1]);

SP.setMetamode(0);
%--- Loop over all frames ---%
for ITER = (myworker+1):nstep:ntotal
    dataframe = SP.setFrame(ITER); 

% FIXME this fails in parallel horribly...
    stepnums(ITER) = sum(dataframe.time.history);

    writeEnsightDatafiles(outBasename, ITER-1, dataframe, varset, reverseIndexOrder);
    if ITER == ntotal
        SP.setMetamode(0);
        writeEnsightMasterFiles(outBasename, range, SP, varset, timeNormalization, reverseIndexOrder);
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

