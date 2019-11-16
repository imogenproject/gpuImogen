function badlist = RHD_cleanup_4d(dirlist, max, start)

badlist = {};
nbad = 1;

if nargin < 1; dirlist = dir('RAD*'); end
if nargin < 2; max = numel(dirlist); end
if nargin < 3; start = 1; end

if isa(dirlist, 'struct')
    j = cell([numel(dirlist) 1]);
    for Q = 1:numel(dirlist); j{Q} = dirlist(Q).name; end
    dirlist = j;
end

for Q = start:max
    fprintf('Run %i | %s | ', int32(Q), dirlist{Q});
    cd(dirlist{Q});
    
    delta = 0;
    
    if exist('4D_XYZT.mat','file')
        try
            load('4D_XYZT.mat', 'F');
        catch ohdear
            fprintf('4D_XYZT exists but load failed. skipping.\n');
            badlist{nbad} = dirlist{Q};
            nbad = nbad+1;
            cd ..; continue;
        end
    else
        fprintf('| ... No 4D_XYZT? trying to load/create ');
        try
            F = util_LoadEntireSpacetime();
            F = DataFrame(F);
            save('4D_XYZT','F','-v7.3');
            fprintf('| Worked. continuing.\n');
            cd ..; continue;
        catch ohffs
            fprintf('| Load unsuccessful. skipping.\n');
            badlist{nbad} = dirlist{Q};
            nbad = nbad+1;
            cd ..; continue;
        end
        
    end
    
    % This is 100% reliable and doesn't warrant a backup
    if ~isa(F, 'DataFrame')
        fprintf('struct -> DataFrame cvt\n');
        F = DataFrame(F);
        save('4D_XYZT.mat','F','-v7.3');
    end
    
    if F.checkForBadRestartFrame(1); delta = 1; end
    if F.chopOutAnomalousTimestep(); delta = 1; end
    
    if delta
        save('4D_XYZTnew.mat','F','-v7.3'); 
    end
    cd ..;
end

ndone = 1 + max - start;
fprintf('Trawled through %i directories.\n', int32(ndone));
if nbad > 0
    fprintf('Encountered %i serious problems:\n', int32(nbad));
    badlist = reshape(badlist, [numel(badlist) 1]);
    disp(badlist);
end


end
