function TF = util_checkFrameExistence(basename, frameset)
% util_checkFrameExistence('basename', [set:of:frames])
% looks, in the PWD, for files of the form
% sprintf('%s_rank%i_%0*i.nc', basename,rank, frameno) for all
% frameno in frameset,
% It returns a true-false matrix the same size as frameset. All files
% which exist are true, all which do not exist are false.

% Initially assume all frames exist
TF = ones(size(frameset));

for x = 1:numel(frameset)
    frameno = frameset(x);
    
    [ftype, ~] = util_FindSegmentFile(basename, 0, frameno);

    if ftype < 0; TF(x) = 0; end
end

end
