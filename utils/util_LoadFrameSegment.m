function dataframe = util_LoadFrameSegment(basename, rank, frameno)
% function dataframe = util_LoadFrameSegment(basename, rank, frameno)

[act, fname] = util_FindSegmentFile(basename, rank, frameno);

if act < 0; error('No .mat or .nc with basename %s, rank %i, frame #%i found in pwd=''%s''\n', basename, rank, frameno, pwd()); end


% Load the next frame into workspace
try
    if act == 1; dataframe = util_NCD2Frame(fname); end
    if act == 2
        tempname = load(fname);
        nom_de_plume = fieldnames(tempname);
        dataframe = tempname.(nom_de_plume{1});
    end
catch ERR
    str = sprintf('SERIOUS on %s: Frame in cwd %s exists but load returned error:\n', getenv('HOSTNAME'), pwd());
    prettyprintException(ERR, 0, str);
    
    dataframe = -1;
end

end
