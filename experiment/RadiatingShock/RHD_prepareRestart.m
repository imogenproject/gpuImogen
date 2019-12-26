function badlist = RHD_prepareRestart(dirlist, max, start, autoframes)
% RHD_prepareRestart({'list','of','directories'}, max, start, autoframes)
% Provides a graphic of completed run status & assists the user in generating the long lists of
% directory names and frame numbers required to resume large numbers of shock runs, processed from 
% dirlist{start} to dirlist{max}, if given, 1:end by default.
% if autoframes is given, it is as if 'autoframes' were entered for all frame count requests
s1 = 'dirs    = {';
s2 = 'frameno = [';
s3 = 'runto   = [';
s4 = 'scalerate=[';

plist = 'plist = [';
badlist = {};
nbad = 1;

dbstop in RHD_prepareRestart.m at 170
dbstop in RHD_prepareRestart.m at 181

if nargin < 2; max = numel(dirlist); end
if nargin < 3; start = 1; end

owd = pwd();
cd('~/gpuimogen/run');
try
    qq = load('run_plist.mat','plist');
    
    haveplist = qq.plist;
    % world's worst hash
    haveplist(:,5) = 100*haveplist(:,1) + haveplist(:,2);
catch
    haveplist = [];
end
cd(owd)


if isa(dirlist, 'struct')
    j = cell([numel(dirlist) 1]);
    for Q = 1:numel(dirlist); j{Q} = dirlist(Q).name; end
    dirlist = j;
end

moveto = [];
tf = input('Move unresumable runs? ');
if tf
    moveto = input('To where: ','s');
end

makeplist = input('Generate plist = [] incl fallback rate for unresumable runs? ');

if makeplist
    disp('Enter -1 as frames to add to force a resumable run to be restarted cold'); 
end

for Q = start:max
    fprintf('Run %i/%i | %s | ', int32(Q), int32(max), dirlist{Q});
    cd(dirlist{Q});
        
    rp = RHD_utils.parseDirectoryName();
    rhash = 100*rp.m + rp.theta;
    if rp.gamma == 167; rp.gam = 1; end
    if rp.gamma == 140; rp.gam = 2; end
    if rp.gamma == 129; rp.gam = 3; end
    if any( (rhash == haveplist(:,5)) & (rp.gam == haveplist(:,4)) )
        fprintf('Run already scheduled for cold rerun: skipping\n');
        cd ..;
        continue;
    end
    
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
    %bot = basepos(1);
    
    N = RHD_utils.lastValidFrame(F, x);
    xmax = size(F.mass,1)*F.dGrid{1};
    
    % Can't resume if it's already crashed into the bottom of the grid
    if N < size(F.mass,4)
        fprintf('Run %i unrestartable, shock off grid at frame %i.\n', Q, int32(N));
        if makeplist
            rp = RHD_utils.parseDirectoryName(pwd());
            
            zz = load('SimInitializer_rank0.mat','IC');
            if isfield(zz.IC.ini, 'fallbackBoost')
                zz = zz.IC.ini.fallbackBoost;
            else
                zz = 0;
            end
            
            [~, vfall] = RHD_utils.extractFallback(x(nlpoint:endpt), F.time.time(nlpoint:endpt)');
            
            switch rp.gamma
                case 167; rt = 1;
                case 140; rt = 2;
                case 129; rt = 3;
            end
            plist = sprintf('%s %.3f, %.2f, %.5g, %i; ', plist, rp.m, rp.theta, vfall + zz, int32(rt));
        end
        cd ..;
        if ~isempty(moveto)
            s = sprintf('!mv %s %s', dirlist{Q}, moveto);
            disp(s);
            eval(s);
        end
        badlist{nbad} = dirlist{Q};
        nbad = nbad + 1;
        continue;
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
    if nargin < 4
        nadd = util_inputNumberBulletproof('Number of frames to add? ');
    else
        nadd = autoframes;
    end
    
    if nadd > 0
        % Previous concatFrame implementation failed to update F.iteration / F.iterMax
        % This is fixed going forward but we need to find out the dumb way
        
        % to be safe
        !rm savefileIndex.mat
        try
            SP = SavefilePortal;
            SP.setFrametype('XY');
            SP.setMetamode(1);
            J = SP.jumpToLastFrame();
        catch
            disp('ERROR: something has shitted up a savefile and the SavefilePortal dumped')
            disp('This needs to be fixed and J must exist & be the last frame before dbcont.');
        end
        
        % there -SHOULDN'T- be a restart Xmillion + 1 'glitch' frame at the end, but
        % stupider crap has happened
        if abs(J.iter - F.time.iteration) > 1
            stuff = load('SimInitializer_rank0.mat');
            spf = stuff.IC.ini.iterMax * stuff.IC.ini.ppSave.dim3 / 100;
            stepsFromFramect = spf * (size(F.mass,4)-1);
            
            if SP.iterWasSaved(stepsFromFramect)
                fprintf('Based on (# frames) x steps/frame we have %i steps\n', stepsFromFramect);
            end
            
            warning('WARNING DANGER DANGER iterations from filenames and F.time.iteration disagree')
            warning('dbstop in RHD_prepareRestart.m at 181 - to access the screwup at the detect point & fix it')
        end
        
        if abs(J.iter - F.time.iteration) < 2
        
            spf = round(J.iter / (size(F.mass, 4)-1));
            
            s1 = sprintf('%s''%s'', ', s1, dirlist{Q});
            s2 = sprintf('%s%i, ', s2, int32(F.time.iteration));
            s3 = sprintf('%s%i, ', s3, int32(J.iter + spf * nadd));
            
            inidt = diff(F.time.time(2:20));
            if std(inidt) > mean(inidt)
                warning('EXTREME DANGER WARNING Simulation save history is SCREWED and will probably OOM/ENODISK this machine if run');
            end
            srate = mean(inidt) / mean(diff(F.time.time((end-20):end)));
            if round(srate) > 1
                fprintf('Inserting saverate scaling of %i.\n', round(srate));
            end
            s4 = sprintf('%s%i, ', s4, int32(round(srate)));
        end
    else
        if nadd == -1
            if makeplist
                rp = RHD_utils.parseDirectoryName(pwd());
                
                zz = load('SimInitializer_rank0.mat','IC');
                if isfield(zz.IC.ini, 'fallbackBoost')
                    zz = zz.IC.ini.fallbackBoost;
                else
                    zz = 0;
                end
                
                [~, vfall] = RHD_utils.extractFallback(x(nlpoint:endpt), F.time.time(nlpoint:endpt)');

                thermotype = 0;
                switch rp.gamma
                    case 167; thermotype = 1;
                    case 140; thermotype = 2;
                    case 129; thermotype = 3;
                end
                plist = sprintf('%s %.3f, %.2f, %.5g, %i; ', plist, rp.m, rp.theta, vfall + zz, thermotype);
            end
            if ~isempty(moveto)
                s = sprintf('!mv %s %s', dirlist{Q}, moveto);
                disp(s);
                eval(s);
            end
        end
        badlist{end+1} = dirlist{Q};
    end
    cd ..;
end

s1 = s1(1:(end-2)); s1 = [s1 '};'];
s2 = s2(1:(end-2)); s2 = [s2 '];'];
s3 = s3(1:(end-2)); s3 = [s3 '];'];
s4 = s4(1:(end-2)); s4 = [s4 '];'];

disp('=========== Paste for special_Resume:');
disp(s1);
disp(s2);
disp(s3);
disp(s4);
disp('runners = 4;');
disp('save(''rst_params.mat'',''dirPrefix'',''dirs'',''frameno'',''runto'',''scalerate'',''runners'')');
disp('=====================================');

if makeplist
    disp('=========== Parameter list for unresumable runs:');
    plist((end-1):end) = '];';
    disp(plist);
end

end
