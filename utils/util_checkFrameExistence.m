function TF = util_CheckFrameExistence(basename, padsize, frameset)

rank = 0;
TF = ones(size(frameset));

for x = 1:numel(frameset);
    frameno = frameset(x);

    if frameno == 0; f1 = sprintf('%s_rank%i_START.nc', basename,rank);
                     f2 = sprintf('%s_rank%i_START.mat',basename,rank);
    else
                     f1 = sprintf('%s_rank%i_%0*i.nc', basename,rank, padsize, frameno);
                     f2 = sprintf('%s_rank%i_%0*i.mat',basename,rank, padsize, frameno);
    end

    if exist(f1, 'file') == 0;
        if exist(f2, 'file') == 0
            % try end
            f3 = sprintf('%s_rank%i_FINAL.nc', basename,rank);
            f4 = sprintf('%s_rank%i_FINAL.mat', basename,rank);
            if exist(f3,'file') == 0
                if exist(f4,'file') == 0
                    TF(x) = 0
                else
                end
            else
            end
        else
        end
    else
    end

end

end
