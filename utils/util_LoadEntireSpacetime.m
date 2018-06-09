function F = util_LoadEntireSpacetime(fields, prefix, hugeStructIsOkay)
% F = util_LoadEntireSpacetime(fields, prefix, hugeStructIsOkay) will replace the
% spatial mass/momentum/energy fields with a 4th index for time.
% Even if loading 2- or 1D data, time is the 4th index.
% If F will require more than 10GB of memory and the third argument is not
% present, this function will require interactive input.
% FIXME: 'fields' is presently nonfunctional and this just returns rho/mom/E

list = enumerateSavefiles();

F = [];
fourd = cell(5);

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

for x = 1:N
    fi = util_LoadWholeFrame(prefix,list.misc.padlen,frameset(x));

    if x == 1 % Set all the default fields in F from the 1st frame
        F = fi;
        % Let's not be *too* casual about larger datasets
        if (numel(F.mass) * 40 * N > 10e9)
            if nargin == 1
                damnTheTorpedoes = input('Yikes! This function will attempt to load over 10GB of data and no override was given. Nonzero if you are sure: ');
            else
                damnTheTorpedoes = hugeStructIsOkay;
            end
            if damnTheTorpedoes == 0; return; end
        end

        fourd{1} = zeros([size(fi.mass,1) size(fi.mass,2) size(fi.mass,3) N]);
        fourd{2} = zeros([size(fi.mass,1) size(fi.mass,2) size(fi.mass,3) N]);
        fourd{3} = zeros([size(fi.mass,1) size(fi.mass,2) size(fi.mass,3) N]);
        fourd{4} = zeros([size(fi.mass,1) size(fi.mass,2) size(fi.mass,3) N]);
        fourd{5} = zeros([size(fi.mass,1) size(fi.mass,2) size(fi.mass,3) N]);
    end
   
    for a = 1:5
        fourd{a}(:,:,:,x) = fi.(fields{a});
    end 
    
    % Final events 
    if x == N
        F.mass = fourd{1};
        F.momX = fourd{2};
        F.momY = fourd{3};
        F.momZ = fourd{4};
        F.ener = fourd{5};
        F.time = fi.time; % Set time history by last frame
        tau = [0; cumsum(F.time.history)];
        F.tFrame = tau(frameset+1);
    end
fprintf('.');

end
fprintf('\n');


end
