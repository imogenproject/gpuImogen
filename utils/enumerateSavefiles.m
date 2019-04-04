function dex = enumerateSavefiles(indir)
% enumerateSavefiles(indir) returns an index of Imogen runfiles saved in
% 'indir'. If no indir is given, the pwd is used.

d0 = pwd(); % Avoid corrupting global state
if nargin > 0; cd(indir); end

if numel(dir('savefileIndex.mat')) == 1
    load('savefileIndex.mat','dex');
    if dex.misc.final == 1 %#ok<NODEF>
        disp('Found existing savefileIndex.mat file; It claims to be complete. Using...');
        return;
    end
end


% Check if this resembles an Imogen savefile directory at all
% If these don't exist, emit warning
basicfiles = numel(dir('ini_settings.mat'));
basicfiles = basicfiles + numel(dir('SimInitializer_rank0.mat'));
if basicfiles ~= 2
    disp('Not seeing @ least 1 of ini_settings.mat or SimInitializer_rank0.mat. This can''t be a valid Imogen run directory.');
    dex = [];
    return;
end

dex = struct('X',[],'Y',[],'Z',[],'XY',[],'XZ',[],'YZ',[],'XYZ',[],'misc',[]);

% Scrape some info out of this...
load('ini_settings.mat','ini');
format = ini.saveFormat;

% All the savefiles that Imogen generates
fieldlist={'X','Y','Z','XY','XZ','YZ','XYZ'};
typelist = {'1D_X','1D_Y','1D_Z','2D_XY','2D_XZ','2D_YZ','3D_XYZ'};
prefixlen=[12, 12, 12, 13, 13, 13, 14];

runComplete = 0;
padLen = 0;

for N = 1:numel(typelist)
    % Per type, list all the files of that type written by rank 0
    if format == ENUM.FORMAT_MAT
        f = dir([typelist{N} '_rank0_*mat']);
    elseif format == ENUM.FORMAT_NC
        f = dir([typelist{N} '_rank0_*nc']);
    elseif format == ENUM.FORMAT_HDF
        f = dir([typelist{N} '_rank0_*h5']);
    end

    flist = [];
    for u = 1:numel(f)
        if f(u).name(prefixlen(N)) == 'S'
            a = 0;
        elseif f(u).name(prefixlen(N)) == 'F'
            X = util_LoadFrameSegment(typelist{N}, 0, 999999, 'metaonly'); % Load last frame
            a = X.iter;
            runComplete = 1; % if a _FINAL exists, the run's done and this index can be saved
            % in order to not waste time on future directory enumerations, which may be slow.
        else
            a = sscanf(f(u).name(prefixlen(N):end),'%d');

            X = util_LoadFrameSegment(typelist{N}, 0, a, 'metaonly'); % l

            if a == X.time.iterMax; runComplete = 1; end

            if format == ENUM.FORMAT_MAT
                padLen = numel(f(u).name) - prefixlen(N) - 3;
            else
                padLen = numel(f(u).name) - prefixlen(N) - 2;
            end
        end
        flist(end+1) = a;
    end
    
    % Append this info to the index
    dex.(fieldlist{N}) = sort(flist);
    %dex = setfield(dex,fieldlist{N},sort(flist));
end

aboot = [];

if runComplete == 1; aboot.final = 1; else; aboot.final = 0; end
aboot.padlen = padLen;
dex.('misc') = aboot;
%dex = setfield(dex, 'misc',aboot);
if mpi_isinitialized()
    if mpi_amirank0(); save('savefileIndex.mat','dex'); end
else
    save('savefileIndex.mat','dex');
end

cd(d0);

end
