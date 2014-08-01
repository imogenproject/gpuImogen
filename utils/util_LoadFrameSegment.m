function dataframe = util_LoadFrameSegment(namebase, padsize, rank, frameno)

    if frameno == 0; f1 = sprintf('%s_rank%i_START.nc', namebase,rank);
                     f2 = sprintf('%s_rank%i_START.mat',namebase,rank);
    else
                     f1 = sprintf('%s_rank%i_%0*i.nc', namebase,rank, padsize, frameno);
                     f2 = sprintf('%s_rank%i_%0*i.mat',namebase,rank, padsize, frameno);
    end

    if exist(f1, 'file') == 0;
        if exist(f2, 'file') == 0
            % try end
            f3 = sprintf('%s_rank%i_FINAL.nc', namebase,rank);
            f4 = sprintf('%s_rank%i_FINAL.mat', namebase,rank);
            if exist(f3,'file') == 0
                if exist(f4,'file') == 0
                    % error
                else
                    % load final.mat
                    fname = f4;
                    act = 2;
                end
            else
                % load final.nc
                fname = f3;
                act = 1;
            end
        else
            % load .mat
            fname = f2;
            act = 2;
        end
    else
        % load .nc
        fname = f1;
        act = 1;
    end

    % Load the next frame into workspace
    try
        if act == 1; dataframe = util_NCD2Frame(fname); end
        if act == 2;
            tempname = load(fname);
            nom_de_plume = fieldnames(tempname);
            dataframe = getfield(tempname,nom_de_plume{1});
        end
    catch ERR
        fprintf('SERIOUS on %s: Frame in cwd %s exists but load returned error:\n', getenv('HOSTNAME'), pwd());
        ERR
        dataframe = -1;
    end

end
