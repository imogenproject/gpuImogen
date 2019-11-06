function RHD_fixconcat(dirlist)
    
if nargin < 1; dirlist = dir('RAD*'); end

n = numel(dirlist);

fprintf('Have %i entries in cwd to fix... ', int32(n));

for Q = 1:n
    if isa(dirlist, 'cell')
        cd(dirlist{Q});
    else
        cd(dirlist(Q).name);
    end
    fprintf('%i ', int32(Q));
    if exist('4D_XYZT.mat','file')
        load('4D_XYZT.mat');
    else
        continue;
    end

    % Previous concatFrame implementation failed to update F.iteration / F.iterMax
    % This is fixed going forward but we need to fix existing runs
    hl = dir('2D_XY*h5');
    if numel(hl) > 0
        % the last savefile names the iteration count
        hl = hl(end).name;
        actualIters = sscanf(hl, '2D_XY_rank0_%i.h5');
        
        if actualIters ~= F.time.iteration
            fprintf('Fixing... ');
            F.time.iteration = actualIters;
            F.time.iterMax = actualIters;
            
            save('4D_XYZT','F');
        end
        
    else
        fprintf('In directory %s - no 2D_XY files? continuing...\n', pwd());
    end
        cd ..;
end

end
