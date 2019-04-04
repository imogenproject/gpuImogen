function exportMerge(SP, varfmt)
% function exportMerge(SP, varfmt = 'default')
% Reads ALL frames of the current type set in SavefilePortal SP and
% merges multi-rank output from parallel simulations into single files
% WARNING: THIS OPERATION MAY REQUIRE AN EXTREMELY LARGE AMOUNT OF MEMORY

if nargin < 2; varfmt = 'default'; end

% Find out what we're eading
SP.setMetamode(1);

x = SP.setFrame(1);

filePrefix = SP.getFilenamePrefix();
padlength = ceil(log10(x.time.iterMax + .5));

ncells = prod(x.parallel.globalDims);
needmem = 8*ncells*10;

fprintf('WARNING: This operation will require somewhat more than %.3dGB of memory.\n', needmem*2^-30);

% FIXME support other forms of output waaaugh
fformat = sprintf('%s_%%0%ii.h5', filePrefix, padlength);

SP.setMetamode(0);
SP.setVarFormat(varfmt);

for q = 1:SP.numFrames
    fprintf('Loading frame %d/%d\n', q, SP.numFrames);
    F = SP.setFrame(q);

    fprintf('Writing frame %d/%d\n', q, SP.numFrames);
    nom = sprintf(fformat, F.time.iteration);
    util_Frame2HDF(nom, F);
    
    clear F; % make sure we never have a 2nd copy of F floating around when we hit the net load!
end

end

