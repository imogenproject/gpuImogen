function badlist = RHD_prepareRestart(dirlist)
% RHD_prepareRestart({'list','of','directories'})
% Provides a graphic of completed run status & assists the user in generating the long lists of
% directory names and frame numbers required to resume large numbers of shock runs
s1 = 'dirs    = {';
s2 = 'frameno = [';
s3 = 'runto   = [';

badlist = cell([1 numel(dirlist)]);
nbad = 1;

if isa(dirlist, 'struct')
    j = cell([numel(dirlist) 1]);
    for Q = 1:numel(dirlist); j{Q} = dirlist(Q).name; end
    dirlist = j;
end

for Q = 1:numel(dirlist)
    cd(dirlist{Q});
    
    try
        load('4D_XYZT.mat', 'F');
    catch ohdear
        pwd
        disp('Unable to load F from 4D_XYZT: not even completed originally? skipping.');
        badlist{nbad} = dirlist{Q};
        nbad = nbad+1;
        cd ..; continue;
    end
    
    autoAnalyze = 1;
    try
        load('autovars.mat', 'autovars');
        
        autovars = double(autovars); % wtf? true, but this somehow ended up as a single once
        nlpoint = autovars(4);
        endpt = autovars(5);
    catch crap
        disp('Problem: Couldn''t load autovars. Restarting a run that hasn''t been analyzed?');
        autoAnalyze = 0;
    end
    
    % This really shouldn't happen any more but can't hurt to catch
    if ~isa(F, 'DataFrame')
        disp('Odd, F is not a DataFrame class yet. Converting...');
        F = DataFrame(F);
    end
    
    x = trackFront2(squeeze(F.pressure), (1:size(F.mass,1))*F.dGrid{1}, .5*(F.gamma+1)/(F.gamma-1));
    basepos = RHD_utils.trackColdBoundary(F);
    bot = basepos(1);
    
    N = RHD_utils.lastValidFrame(F, x);
    xmax = size(F.mass,1)*F.dGrid{1};
    
    % Can't resume if it's already crashed into the bottom of the grid
    if N < size(F.mass,4)
        fprintf('Problem: Run %i cannot be restarted, shock has already walked off grid at frame %i.\n', 0, int32(N));
        badlist{nbad} = dirlist{Q};
        nbad = nbad + 1;
        cd ..; continue;
    end
    
    plot(x,'r');
    hold on;
    plot(basepos,'b');
    if autoAnalyze
        plot([nlpoint endpt], x(round([nlpoint endpt])), 'rO');
        fprintf('Original FFTed interval indicated.\n');
    end
    % x max value
    plot([0 size(F.mass,4)], [xmax xmax], 'b-.x');
    plot([0 size(F.mass,4)], [0 0], 'r-.x');
    hold off;
    
    fprintf('%i frames: ', int32(size(F.mass, 4)-1));
    nadd = input('Number of frames to add? ');
    
    if nadd > 0
        % Previous concatFrame implementation failed to update F.iteration / F.iterMax
        % This is fixed going forward but we need to find out the dumb way
        hl = dir('2D_XY*h5');
        % the last savefile names the iteration count
        hl = hl(end).name;
        actualIters = sscanf(hl, '2D_XY_rank0_%i.h5');
        
        spf = round(actualIters / (size(F.mass, 4)-1));
        
        s1 = sprintf('%s''%s'', ', s1, dirlist{Q});
        s2 = sprintf('%s%i, ', s2, int32(F.time.iteration));
        s3 = sprintf('%s%i, ', s3, int32(actualIters + spf * nadd));
    else
        badlist{end+1} = dirlist{Q};
    end
    cd ..;
end

s1 = s1(1:(end-2)); s1 = [s1 '};'];
s2 = s2(1:(end-2)); s2 = [s2 '];'];
s3 = s3(1:(end-2)); s3 = [s3 '];'];

disp('=========== Paste for special_Resume: ');
disp(s1);
disp(s2);
disp(s3);

end
