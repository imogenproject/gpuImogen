function TF = util_checkFrameExistence(basename, padsize, frameset)
% util_checkFrameExistence('basename',padsize, [set:of:frames])
% looks, in the PWD, for files of the form
% sprintf('%s_rank%i_%0*i.nc', basename,rank, padsize, frameno) for all
% frameno in frameset,
% It returns a true-false matrix the same size as frameset. All files
% which exist are true, all which do not exist are false.
rank = 0;
% Initially assume all frames exist
TF = ones(size(frameset));

for x = 1:numel(frameset);
    frameno = frameset(x);
    
    % First, just try numerics
    % This is the new default
    f1 = sprintf('%s_rank%i_%0*i.nc', basename,rank, padsize, frameno);
    f2 = sprintf('%s_rank%i_%0*i.mat',basename,rank, padsize, frameno);
    
    if (exist(f1,'file') == 0) && (exist(f2,'file') == 0)
        % Ugh... try _START for frame 0, then in desperation try _END
        if frameno == 0;
            f1 = sprintf('%s_rank%i_START.nc', basename,rank);
            f2 = sprintf('%s_rank%i_START.mat',basename,rank);
        else
            f1 = sprintf('%s_rank%i_FINAL.nc', basename,rank);
            f2 = sprintf('%s_rank%i_FINAL.mat', basename,rank);
        end
        if (exist(f1,'file') == 0) && (exist(f2, 'file') == 0)
            
            TF(x) = 0;
        end
    end

end

end
