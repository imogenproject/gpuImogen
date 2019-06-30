function F = util_LoadEntireSpacetime(fields, prefix, hugeStructIsOkay)
% F = util_LoadEntireSpacetime(fields, prefix, hugeStructIsOkay) will replace the
% spatial mass/momentum/energy fields with a 4th index for time.
% Even if loading 2- or 1D data, time is the 4th index.
% If F will require more than 10GB of memory and the third argument is not
% present, this function will require interactive input.
% FIXME: 'fields' is presently nonfunctional and this just returns rho/mom/E

list = enumerateSavefiles();

F = [];

if nargin == 0
    fields = {'mass','momX','momY','momZ','ener'};
    prefix = '3D_XYZ';
end

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
    
    fprintf('.');
    if mod(x,100) == 0; fprintf('\n'); end

end
fprintf('\n');




end
