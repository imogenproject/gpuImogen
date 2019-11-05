function RHD_prepareRestart(dirlist)
    
s1 = 'dirs    = {';
s2 = 'frameno = [';
s3 = 'runto   = [';

for Q = 1:numel(dirlist)
    cd(dirlist{Q});
    load('4D_XYZT.mat');

    s1 = sprintf('%s''%s'', ', s1, dirlist{1});
    s2 = sprintf('%s%i, ', s2, int32(F.time.iteration));
    
    autoAnalyze = 1;
    
    try
        load('autovars.mat');
        
        autovars = double(autovars); % wtf? true, but this somehow ended up as a single once
        pts = autovars(1:2);
        pointspace = autovars(3);
        nlpoint = autovars(4);
        endpt = autovars(5);
    catch crap
        disp('Problem: Couldn''t load autovars. Restarting a run that hasn''t been analyzed?');
        autoAnalyze = 0;
    end
    
    if ~isa(F, 'DataFrame')
        F = DataFrame(F);
    end
    
    x = trackFront2(squeeze(F.pressure), (1:size(F.mass,1))*F.dGrid{1}, .5*(F.gamma+1)/(F.gamma-1));
    
    % Automatically strip junk at the end of the run off if it hit the end of the grid
    N = RHD_utils.lastValidFrame(F, x);
    
    if N < size(F.mass,4)
        fprintf('Problem: Run %i cannot be restarted, shock has already walked off grid at frame %i.\n', 0, int32(N));
    end
    
    
    plot(x(3:end));
    hold on;
    
    if autoAnalyze
        plot([nlpoint endpt], x(round([nlpoint endpt])), 'rO');
        fprintf('Original FFTed interval indicated.\n');
    end
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
        
        s3 = sprintf('%s%i, ', s3, int32(actualIters + spf * nadd));
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
