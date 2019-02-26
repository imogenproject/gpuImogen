function exportMerge(SP)
% function exportMerge(SP)
% Reads ALL frames of the current type set in SavefilePortal SP and
% merges multi-rank output from parallel simulations into single files
% WARNING: THIS OPERATION MAY REQUIRE AN EXTREMELY LARGE AMOUNT OF MEMORY

% Find out what we're reading
SP.setMetamode(1);

x = SP.setFrame(1);

filePrefix = SP.getFilenamePrefix();
padlength = ceil(log10(x.time.iterMax + .5));

% FIXME support other forms of output waaaugh
fformat = sprintf('%s_%%0%ii.h5', filePrefix, padlength);

SP.setMetamode(0);

for q = 1:SP.numFrames
    F = SP.setFrame(q);

    nom = sprintf(fformat, F.time.iteration);
    util_Frame2HDF(nom, F);
end

end

