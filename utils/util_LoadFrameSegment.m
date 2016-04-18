function dataframe = util_LoadFrameSegment(basename, padsize, rank, frameno)

if isa(basename,'double')
    strnames={'1D_X','1D_Y','1D_Z','2D_XY','2D_XZ','2D_YZ','3D_XYZ'};
    try
        basename = strnames{basename};
    catch MERR
        basename
        error('Numeric basename passed was not an integer [1, 7].');
    end
end

    act = -1;

    % Probably 1 of these
    f1 = sprintf('%s_rank%i_%0*i.nc', basename,rank, padsize, frameno);
    if exist(f1, 'file') == 0;
        f2 = sprintf('%s_rank%i_%0*i.mat',basename,rank, padsize, frameno);
	if exist(f2, 'file') 
            % load .mat
            fname = f2;
            act = 2;
        end
    else
        % load .nc
        fname = f1;
        act = 1;
    end

    if (act < 0) && (frame == 0) % compat w/old versions
    f1 = sprintf('%s_rank%i_START.nc', basename,rank);
    if exist(f1, 'file') == 0;
        f2 = sprintf('%s_rank%i_START.mat',basename,rank);
	if exist(f2, 'file') 
            % load .mat
            fname = f2;
            act = 2;
        end
    else
        % load .nc
        fname = f1;
        act = 1;
    end
    end

    if (act < 0) && (frame > 0)
        % try end
        f3 = sprintf('%s_rank%i_FINAL.nc', basename,rank);
        f4 = sprintf('%s_rank%i_FINAL.mat', basename,rank);
        if exist(f3,'file') == 0
            if exist(f4,'file') 
                % load final.mat
                fname = f4;
                act = 2;
            end
        else
            % load final.nc
            fname = f3;
            act = 1;
        end
    end

    if act < 0; error('No .mat or .nc with basename %s, rank %i, frame %i findable in pwd=''%s''\n', basename, rank, frameno, pwd()); end

    % Load the next frame into workspace
    try
        if act == 1; dataframe = util_NCD2Frame(fname); end
        if act == 2;
            tempname = load(fname);
            nom_de_plume = fieldnames(tempname);
            dataframe = getfield(tempname,nom_de_plume{1});
        end
    catch ERR
        str = sprintf('SERIOUS on %s: Frame in cwd %s exists but load returned error:\n', getenv('HOSTNAME'), pwd());
        prettyprintException(ERR, 0, str);

        dataframe = -1;
    end

end
