function TF = util_checkFrameExistence(prefix, basename, frameset)
% util_checkFrameExistence('prefix', 'basename', [set:of:frames])
% looks, in the PWD, for files of the form
% sprintf('%s%s_rank%i_%0*i.nc', prefix, basename,rank, frameno) for all
% frameno in frameset,
% It returns a true-false matrix the same size as frameset. All files
% which exist are true, all which do not exist are false.

% Initially assume all frames exist
TF = ones(size(frameset));

for x = 1:numel(frameset)
    frameno = frameset(x);
    
    [ftype, ~] = util_FindSegmentFile(prefix, basename, 0, frameno);

    if ftype < 0; TF(x) = 0; end
end

end
