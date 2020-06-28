function F = util_LoadEntireSpacetime(fields, prefix, hugeStructIsOkay, progressReport)
% F = util_LoadEntireSpacetime(fields, prefix, hugeStructIsOkay, progressReport) will replace the
% spatial mass/momentum/energy fields with a 4th index for time.
% Even if loading 2- or 1D data, time is the 4th index.
% fields: FIXME: 'fields' is presently nonfunctional and this just returns rho/mom/E
% prefix: One of 3D_XYZ, 2D_XY, 2D_YZ, 2D_XZ, 1D_X, 1D_Y, 1D_Z; Default 3D_XYZ;
%         This refers to the file set to access
% hugeStructIsOkay: Turns off warning and confirmation request for output predicted to exceed 10GB
% progressReport:   0 - none; 1 - print '.' per frame; 2 - print # per 100 frames; 3 - waitbar()

list = enumerateSavefiles('');

F = [];

if nargin < 4; progressReport = 2; end
if nargin < 3; hugeStructIsOkay = 0; end
if nargin < 2; prefix = '3D_XYZ'; end
if nargin < 1; fields = {'mass','momX','momY','momZ','ener'}; end

if isempty(fields); fields = {'mass','momX','momY','momZ','ener'}; end
if isempty(prefix); prefix = '3D_XYZ'; end

% FIXME: This feels dumb. ENUM a cell array that maps these?
if strcmp(prefix, '3D_XYZ'); frameset = list.XYZ; end
if strcmp(prefix, '2D_XY'); frameset = list.XY; end
if strcmp(prefix, '2D_XZ'); frameset = list.XZ; end
if strcmp(prefix, '2D_YZ'); frameset = list.YZ; end
if strcmp(prefix, '1D_X'); frameset = list.X; end
if strcmp(prefix, '1D_Y'); frameset = list.Y; end
if strcmp(prefix, '1D_Z'); frameset = list.Z; end

N = numel(frameset);

nfields = numel(fields);
fourd = cell(nfields, 1);

if progressReport == 2
    fprintf('%i frames: ', int32(N));
end

if progressReport == 3
    wb = waitbar(0, sprintf('4D load: 0/%i', int32(N)));
end

for x = 1:N
    fi = util_LoadWholeFrame(prefix,frameset(x));

    if x == 1 % Set all the default fields in F from the 1st frame
        F = fi;
        % Let's not be *too* casual about larger datasets
        if (numel(F.mass) * 40 * N > 10e9)
            if nargin < 3
                damnTheTorpedoes = input('Yikes! This function will attempt to load over 10GB of data and no override was given. Nonzero if you are sure: ');
            else
                damnTheTorpedoes = hugeStructIsOkay;
            end
            if damnTheTorpedoes == 0; return; end
        end

        dee = [size(fi.mass,1) size(fi.mass,2) size(fi.mass,3) N];
        for a = 1:nfields
            fourd{a} = zeros(dee);
        end
        
        F.time.time = zeros([N 1]);
    end

    % Build the 4D structure
    for a = 1:nfields
        fourd{a}(:,:,:,x) = fi.(fields{a});
    end 
    
    % Append frame time
    F.time.time(x) = fi.time.time;
    
    % Final events 
    if x == N
        F.mass = fourd{1};
        F.momX = fourd{2};
        F.momY = fourd{3};
        F.momZ = fourd{4};
        F.ener = fourd{5};
        tau = F.time.time;
        
        F.time = fi.time; % Set time history by last frame
        F.time.time = tau;% But copy last frame time
    end
    
    switch progressReport
        case 1
            fprintf('.');
            if mod(x, 100) == 0; fprintf('\n'); end
        case 2
            if mod(x, 100) == 0; fprintf('%i ', int32(x)); end
            if mod(x, 2000) == 0; fprintf('\n'); end
        case 3
            if mod(x, 10) == 0; waitbar(x/N, wb, sprintf('4F load: %i/%i', int32(x), int32(N))); end
    end

end
fprintf('\n');




end
