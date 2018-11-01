function [ftype, fname] = util_FindSegmentFile(basename, rank, frameno)
% [ftype, fname] = util_FindSegmentFile(basename, rank, frameno)

if isa(basename,'double')
    strnames={'1D_X','1D_Y','1D_Z','2D_XY','2D_XZ','2D_YZ','3D_XYZ'};
    try
        basename = strnames{basename};
    catch MERR
        disp('Basename is: ');
        disp(basename);
        error('Numeric basename passed was not an integer [1, 7].');
    end
end

ftype = -1;
fname = '';

% It makes no sense to burden the user, just try pad counts from 1 to
% 8, spanning run.iterMax from 1 to 100 billion. That oughta cover it.
for trypads = 1:11
    padsize = trypads;
    f0 = sprintf('%s_rank%i_%0*i.h5', basename, rank, padsize, frameno);
    if exist(f0, 'file') == 0
        
        f1 = sprintf('%s_rank%i_%0*i.nc', basename,rank, padsize, frameno);
        if exist(f1, 'file') == 0
            f2 = sprintf('%s_rank%i_%0*i.mat',basename,rank, padsize, frameno);
            if exist(f2, 'file')
                % load .mat
                fname = f2;
                ftype = ENUM.FORMAT_MAT;
            end
        else
            % load .nc
            fname = f1;
            ftype = ENUM.FORMAT_NC;
        end
    else
        fname = f0;
        ftype = ENUM.FORMAT_HDF;
    end
    
    if ftype > 0; break; end
    
end

if (ftype < 0) && (frame == 0) % compat w/old versions
    f1 = sprintf('%s_rank%i_START.nc', basename,rank);
    if exist(f1, 'file') == 0
        f2 = sprintf('%s_rank%i_START.mat',basename,rank);
        if exist(f2, 'file')
            % load .mat
            fname = f2;
            ftype = ENUM.FORMAT_MAT;
        end
    else
        % load .nc
        fname = f1;
        ftype = ENUM.FORMAT_NC;
    end
end

if (ftype < 0) && (frame > 0) % One last guess...
    % try end
    f3 = sprintf('%s_rank%i_FINAL.nc', basename,rank);
    f4 = sprintf('%s_rank%i_FINAL.mat', basename,rank);
    if exist(f3,'file') == 0
        if exist(f4,'file')
            % load final.mat
            fname = f4;
            ftype = ENUM.FORMAT_MAT;
        end
    else
        % load final.nc
        fname = f3;
        ftype = ENUM.FORMAT_NC;
    end
end

end
