function dex = enumerateSavefiles(filePrefix, indir)
% enumerateSavefiles(filePrefix, indir) returns an index of Imogen runfiles saved in
% 'indir'. If no indir is given, the pwd is used.

d0 = pwd(); % Avoid corrupting global state
if nargin > 1; cd(indir); end

if numel(dir('savefileIndex.mat')) == 1
    load('savefileIndex.mat','dex');
    if dex.misc.final == 1 %#ok<NODEF>
        disp('Found existing savefileIndex.mat file; It claims to be complete. Using...');
        return;
    end
end

if (strlength(filePrefix) > 0) && (filePrefix(end) ~= '_')
    filePrefix = [filePrefix '_'];
end

% Check if this resembles an Imogen savefile directory at all
% If these don't exist, emit warning
basicfiles = numel(dir('ini_settings.mat'));
basicfiles = basicfiles + numel(dir('SimInitializer_rank0.mat'));
if basicfiles ~= 2
    disp('Not seeing @ least 1 of ini_settings.mat or SimInitializer_rank0.mat. This can''t be a valid Imogen run directory.');
    disp('Will attempt to proceed anyway. Please input 1 2 or 3 for MAT, NC and HDF5 format: ');
    f0 = input('');
    switch f0
        case 1; format = ENUM.FORMAT_MAT;
        case 2; format = ENUM.FORMAT_NC;
        case 3; format = ENUM.FORMAT_HDF;
        otherwise; error('invalid format. aborting.');
    end
else
    % Scrape some info out of this...
    load('ini_settings.mat','ini');
    format = ini.saveFormat;
end

dex = struct('X',[],'Y',[],'Z',[],'XY',[],'XZ',[],'YZ',[],'XYZ',[],'misc',[]);

% All the savefiles that Imogen generates
fieldlist={'X','Y','Z','XY','XZ','YZ','XYZ'};
typelist = {'1D_X','1D_Y','1D_Z','2D_XY','2D_XZ','2D_YZ','3D_XYZ'};
prefixlen=[12, 12, 12, 13, 13, 13, 14] + strlength(filePrefix);

runComplete = 0;
padLen = 0;

for N = 1:numel(typelist)
    
    % Per type, list all the files of that type written by rank 0
    if format == ENUM.FORMAT_MAT
        f = dir([filePrefix typelist{N} '_rank*_*mat']);
    elseif format == ENUM.FORMAT_NC
        f = dir([filePrefix typelist{N} '_rank*_*nc']);
    elseif format == ENUM.FORMAT_HDF
        f = dir([filePrefix typelist{N} '_rank*_*h5']);
    end

    flist = [];
    for u = 1:numel(f)
        c = analyzeSavefileName(f(u).name);
        
        a = sscanf(f(u).name(c.framepos:end),'%d');
c
a
        
        X = util_LoadFrameSegment(filePrefix, typelist{N}, 0, a, 'metaonly'); % l
        
        if a == X.time.iterMax; runComplete = 1; end
        
        if format == ENUM.FORMAT_MAT
            padLen = numel(f(u).name) - prefixlen(N) - 3;
        else
            padLen = numel(f(u).name) - prefixlen(N) - 2;
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

function c = analyzeSavefileName(f)
% need:
% prefix
% type
% rank digits
% frame digits
% extension type

% The generic format is prefix_TYPE_rankXXX_YYY.ext

% find the 1/2/3 of the type
for j = 1:numel(f)
    if f(j) == '_'; break; end
%    if any(f(j) == ['1' '2' '3']); break; end
end
j=j+1;

if j > 2
    p = f(1:(j-2));
    f = f(j:end);
    x = j;
else
    p = '';
    x = 0;
end
p
f
x

t = -1;
typelist = {'1D_X','1D_Y','1D_Z','2D_XY','2D_XZ','2D_YZ','3D_XYZ'};
for n = 1:7
    if strcmp(f(1:numel(typelist{n})), typelist{n}) == 1
        t = n;
        f = f((numel(typelist{n})+ 6):end);
        x = x + numel(typelist{n}) + 5;
        break
    end
end
t
f
x
rankpos = x;

for r = 1:numel(f)
    if f(r) == '_'; break; end
end
r=r-1;
f=f((r+2):end);
framepos = x + r + 1;

for d = 1:numel(f)
    if f(d) == '.'; break; end
end
d=d-1;
f = f((d+2):end);

if strcmp(f, 'mat') == 1; e = 1; end
if strcmp(f, 'nc') == 1; e = 2; end
if strcmp(f, 'h5') == 1; e = 3; end

c = struct('prefix', p, 'type', t, 'rankdig', r, 'rankpos', rankpos, 'framedig', d, 'framepos', framepos, 'ext', e);

end
