 %dirlist = dir('RAD*');

for Q = 1:numel(dirlist)
    if isa(dirlist, 'cell')
        cd(dirlist{Q});
    else
        cd(dirlist(Q).name);
    end
    
    load('4D_XYZT.mat');
    
    if ~isa(F, 'DataFrame')
        F = DataFrame(F);
    end
    
    try
        load('autovars.mat');
        
        autovars = double(autovars); % wtf? true, but this somehow ended up as a single once
        pts = autovars(1:2);
        pointspace = autovars(3);
        nlpoint = autovars(4);
        endpt = autovars(5);
    catch crap
        disp('Problem: Couldn''t load autovars. Run not analyzed. Not checking convergence.');
        cd ..;
        continue;
    end
    
    x = trackFront2(squeeze(F.pressure), (1:size(F.mass,1))*F.dGrid{1}, .5*(F.gamma+1)/(F.gamma-1));

    xmax = size(F.mass,1)*F.dGrid{1};

    basepos = trackBase(F.pressure(:,1,1,1), (1:size(F.mass,1))*F.dGrid{1});
    
xShock = basepos - x(1);

tHat = xShock / F.velX(1,1,1,1);

tNormal = F.time.time / tHat / 2 / pi;

pointspace = 2;
interframe1 = RHD_utils.projectParabolicMinimum(x, pts(1), 1, pointspace);

if ~isreal(interframe1)
    hold off;
    plot((pts(1)-10):(pts(2)+10), x((pts(1)-10):(pts(2)+10)));
    hold on;
    disp('WARNING: the shock bounce period appears to be VERY short. Trying again with tighter spaced points.');
    interframe1 = RHD_utils.projectParabolicMinimum(x, pts(1), 1, 1);
    pointspace = 1;    
end

interframe2 = RHD_utils.projectParabolicMinimum(x, pts(2), 1, pointspace);
if ~isreal(interframe2) && pointspace > 1
    disp('WARNING: the shock bounce period appears to be VERY short. Trying again with tighter spaced points.');
    interframe2 = RHD_utils.projectParabolicMinimum(x, pts(2), 1, 1);
end

% Compute the interval of the demarked oscillation period to find the 'fundamental' period
tfunda = interp1(1:size(tNormal), tNormal, [interframe1 interframe2],'linear');

plot([interframe1 interframe2], x(round([interframe1 interframe2])), 'kv', 'MarkerSize', 10);
hold off;

tfunda0 = tNormal(pts);
ffunda0 = 1/(tfunda0(2) - tfunda0(1));

ffunda = 1/(tfunda(2)-tfunda(1));

dfreq = abs(ffunda/ffunda0 - 1);
if dfreq > .03
    disp(['WARNING: 0th and 2nd order fundamental frequencies differ by ' num2str(100*dfreq) '%... you sure those were bounce points chief?']); 
else
    disp(['2nd order extrapolation of ' num2str(ffunda) ' differs from 0th order estimate by ' num2str(100*dfreq) '%, good.']);
end

% This attempts to get an even number of cycles into the FFT
% this minimizes artificial spectral peaks, sidebanding and junk in general
spacing = round(pts(2)-pts(1));
cycles = round(endpt-nlpoint(1))/(interframe2 - interframe1) - 1;

endpt = RHD_utils.walkdown(x, round(nlpoint(1) + cycles*(interframe2-interframe1)), spacing);
stpt = RHD_utils.walkdown(x, nlpoint(1), spacing);

npt = 1+endpt-stpt;

if npt/2 ~= round(npt/2); stpt = stpt-1; end

% Removes the constant and linear parts of the shock position drift,
% again to minimize spurious spectral junk in the FFT
timepts = tNormal(stpt:endpt);
pospts = x(stpt:endpt)';

[coeffs, resid] = polyfit(timepts, pospts, 1);
fprintf('Shock fallback velocity = %f\n', coeffs(1)/(2*pi*tHat));

oscil = pospts - (coeffs(1)*timepts + coeffs(2));

% Rescale the fft
xfourier = 2*abs(fft(oscil))/numel(oscil);
xi = numel(xfourier(2:end/2));

runparams = RHD_utils.parseDirectoryName(); % the worst way of learning this lol
rth = runparams.theta;

% This is the time unit for display of FFT results
tfunda = (timepts(end) - timepts(1));

  
    

    % Plot shock X
    subplot(2,1,1);
    plot(x(3:end));
    hold on;
    plot([nlpoint endpt], x(round([nlpoint endpt])), 'rO');
    if any(x + xShock > .8*xmax)
        plot([1 size(F.mass,4)], [xmax xmax], 'r-');
    end
    if any(x < 10*F.dGrid{1})
        plot([1 size(F.mass,4)], [0 0], 'g-');
    end
    hold off;
    
    % Plot fourier spectrogram
    subplot(2,1,2);
    hold off;
    plot((1:xi)/tfunda, xfourier(2:end/2)/xShock,'b-');
    xlim([0 7]);
    
    clvl = input('Indicate convergence level (1-5): ');
    
    runparams = RHD_utils.parseDirectoryName();
    
    if runparams.gamma == 167
        if exist('f53','var')
            f53.updateConvergenceLevel(runparams.m, runparams.theta, clvl);
        elseif exist('self', 'var')
            
        else
            disp('Access to FMHandler directly is required to insert data.\n');
        end
    elseif runparams.gamma == 140
        if exist('f75','var')
            f75.updateConvergenceLevel(runparams.m, runparams.theta, clvl);
        else
            disp('Access to FMHandler directly is required to insert data.\n');
        end
    elseif runparams.gamma == 129
        if exist('f97','var')
            f97.updateConvergenceLevel(runparams.m, runparams.theta, clvl);
        else
            disp('Access to FMHandler directly is required to insert data.\n');
        end
    else
        disp('Strange, an adiabatic index that is not 5/3, 7/5 or 9/7?\n');
    end
    
    cd ..;
end


