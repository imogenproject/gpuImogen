function dataframe = util_LoadFrameSegment(basename, rank, frameno, meta)
% function dataframe = util_LoadFrameSegment(basename, rank, frameno, metaonly)
% > basename: prefix of files (2D_XY, 3D_XYZ, etc)
% > rank:     integer of MPI rank whose segment we load
% > frameno:  frame number of file to load
% > meta:     If identical the to the string 'metaonly', does not load the large 3D data arrays

if nargin < 4; meta = 'everything'; end

[act, fname] = util_FindSegmentFile(basename, rank, frameno);

if act < 0; error('No .mat or .nc with basename %s, rank %i, frame #%i found in pwd=''%s''\n', basename, rank, frameno, pwd()); end

% Load the next frame into workspace
try
    if act == ENUM.FORMAT_NC; dataframe = util_NCD2Frame(fname, meta); end
    if act == ENUM.FORMAT_MAT
        tempname = load(fname);
        nom_de_plume = fieldnames(tempname);
        dataframe = tempname.(nom_de_plume{1});
        if strcmp(meta, 'metaonly')
            if isfield(dataframe, 'momX'); dataframe.varFormat = 'conservative'; else; dataframe.varFormat = 'primitive'; end
            if isfield(dataframe, 'mass2'); dataframe.twoFluids = 1; else; dataframe.twoFluids = 0; end
        end
    end
    if act == ENUM.FORMAT_HDF; dataframe = util_HDF2Frame(fname, meta); end
catch ERR
    str = sprintf('SERIOUS on %s: Frame %s in cwd %s exists but load returned error\nThis most likely occurs due to out-of-disk while writing: Check file sizes & delete truncated garbage.', getenv('HOSTNAME'),fname,  pwd());
    prettyprintException(ERR, 0, str);
    
    dataframe = -1;
end

end
