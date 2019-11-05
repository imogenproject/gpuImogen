%function result = RHD_1danalysis(F, automode)
%
%if nargin == 0
%    disp('Did not receive dataframe: Attempting to load');
    load 4D_XYZT
%end

if exist('rhdAutodrive','var')
    try
        load('autovars.mat');
        
        autovars = double(autovars); % wtf? true, but this somehow ended up as a single one
        pts = autovars(1:2);
        pointspace = autovars(3);
        nlpoint = autovars(4);
        endpt = autovars(5);
    catch crap
        disp('Problem: Couldn''t load autovars. Manual analysis required.');
        reenableAuto = 1;
        clear rhdAutodrive;
    end
end

if ~isa(F, 'DataFrame')
    F = DataFrame(F);
end

x = trackFront2(squeeze(F.pressure), (1:size(F.mass,1))*F.dGrid{1}, .5*(F.gamma+1)/(F.gamma-1));

basepos = trackBase(F.pressure(:,1,1,1), (1:size(F.mass,1))*F.dGrid{1});

xShock = basepos - x(1);

tHat = xShock / F.velX(1,1,1,1);

tNormal = F.time.time / tHat / 2 / pi;

% Automatically strip junk at the end of the run off if it hit the end of the grid
N = RHD_utils.lastValidFrame(F, x);

if N < size(F.mass,4)
    fprintf('Note: shock gets too close to end of grid at frame %i: Truncating dataset.\n', N);
    %F.truncate([], [], [], 1:N);
    %x=x(1:N);
end

hold off;
% throw up pretty spacetime diagram
dlrdt = diff(squeeze(F.mass), 1, 2)./squeeze(F.mass(:,1,1,1:(end-1)));
imagesc(dlrdt);
thefig = gca();

% This is usually a reasonable automatic scaling of the displayed plot
h0 = round(numel(x)/2);
q = dlrdt(round(x(h0)/F.dGrid{1}+250), (h0-300):(h0+300));
q = max(abs(q));
thefig.CLim = [-2*q, 2*q];

% Acquire data points demarking an oscillation period
% This is no longer needed for mode identification but it is useful for making the
% spectrograms of pure tones awful purty
if exist('rhdAutodrive','var') == 0
    fprintf('Please zoom in & click two points demarking one oscillation period.\n');
    P = pointGetter();
    pt = P.waitForClick();
    P.clickedPoint = [];
    pts(1) = RHD_utils.walkdown(x, round(pt(1)), 20);
    fprintf('First point: %i\n', pts(1));
    
    pt = P.waitForClick();
    P.clickedPoint = [];
    pts(2) = RHD_utils.walkdown(x, round(pt(1)), 20);
    fprintf('Second point: %i\n', pts(2));
    
    % In case they click right then left
    pts = sort(pts);
else
    fprintf('First point: %i\n', pts(1));
    fprintf('Second point: %i\n', pts(2));
end

% This block projects parabolas onto the left and right areas indicated
% and takes their nearby intercept as the true shock bounce point;
% This is much better than the nearest-guess method previously used.
cla('reset');

plot((pts(1)-10):(pts(2)+10), x((pts(1)-10):(pts(2)+10)));
hold on;

if exist('rhdAutodrive','var') == 0
    pointspace = input('Enter value to force pointspace or blank for default=2: ');
end
if isempty(pointspace); pointspace = 2; end

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

if ~exist('rhdAutodrive','var')
    input('Enter if satisfied with projection: ');
end

% Have the user enter the interval to fourier transform
% One point uses [point end]
% Two may be used if the shock drifts off the simulation volume
plot(x(3:end));

if exist('rhdAutodrive','var') == 0
    disp('Click the start & end points of the interval to transform.');
    
    P = pointGetter();
    pt = P.waitForClick();
    P.clickedPoint = [];
    nlpoint(1) = round(pt(1));
    fprintf('Starting at: %i\n', pt(1));
    
    pt = P.waitForClick();
    P.clickedPoint = [];
    nlpoint(2) = round(pt(1));
    fprintf('Ending at: %i\n', pt(1));
   
else
    nlpoint(2) = endpt;
    if rhdAutodrive == 2
        disp('Click the start & end points of the interval to transform.');
        
        P = pointGetter();
        pt = P.waitForClick();
        P.clickedPoint = [];
        nlpoint(1) = round(pt(1));
        fprintf('Starting at: %i\n', pt(1));
        
        pt = P.waitForClick();
        P.clickedPoint = [];
        nlpoint(2) = round(pt(1));
        fprintf('Ending at: %i\n', pt(1));
    end
end

% In case they click right then left
nlpoint = sort(nlpoint);
    
endpt = nlpoint(2);

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

% Throw up fft of shock position
hold off;
plot((1:xi)/tfunda, xfourier(2:end/2)/xShock,'b-');
hold on

% Compute the luminosity on the given interval and normalize it by L(t=0) and fft it
rr = RHD_utils.computeRelativeLuminosity(F, rth);
rft = 2*fft(rr(stpt:endpt))' / (1+endpt-stpt);

plot((1:xi)/tfunda,abs(rft(2:end/2)), 'r-');

lpp = max(rr(stpt:endpt))-min(rr(stpt:endpt));

% helpers to identify gaussian peaks
ispeak = @(y, xi) (y(xi) > y(xi+1)) & (y(xi) > y(xi-1)) & ( y(xi) > 2*sum(y(xi+[-2, -1, 1, 2])) );
isgausspeak = @(mag, center, std) (abs(center) < .55) & (std < 3);

grid on;
fatdot = [];
raddot = [];
possibleFmode = 0;

xresid = xfourier(2:end/2);
for n = 1:10
    % Pick the highest peak
    [~, pkidx] = max(xresid);
    p = pkidx + 1;
    
    if p < 5; continue; end
    if p + 4 > numel(xresid); break; end
    
    [mag, center, std] = RHD_utils.gaussianPeakFit(xfourier, p);
    
    if isgausspeak(mag, center - p, std)
        fatdot(end+1,:) = [center-1, mag, std];
        
        % extract the radiance spectral peak as well
        [mag, center, std] = RHD_utils.gaussianPeakFit(abs(rft), p);
        raddot(end+1,:) = [center-1, mag, std];
    end
    
    % Truncate the entire peak
    xresid = RHD_utils.chopPeakForSearch(xresid, pkidx);
end

spec_residual = RHD_utils.subtractKnownPeaks(xfourier(2:end/2), fatdot);

% convert the indexes in fatdot to frequencies
fatdot(:,1) = fatdot(:,1) / tfunda;
raddot(:,1) = raddot(:,1) / tfunda;

% Drop the detected peaks onto the graph and print about them
plot(fatdot(:,1), fatdot(:,2)/xShock, 'kv', 'MarkerSize', 8);
xlim([0 2*max(fatdot(:,1))]);

fprintf("Frequency resolution = +-%f\n", 1/tfunda);
modenames = cell([n 1]);
for n = 1:size(fatdot,1)
    fi = fatdot(n,1);
    
    if ~isempty(modenames{n}); continue; end
    
    s0 = RHD_utils.assignModeName(fi, runparams.m, runparams.gamma, runparams.theta);

    if numel(s0) > 0
        modenames{n} = s0;
        % Attempt to identify harmonic distortion
        for u=1:size(fatdot,1)
            if isempty(modenames{u})
                for hd = 2:9
                    if abs(fatdot(u,1) - hd*fatdot(n,1)) < 1.5/tfunda
                        modenames{u} = sprintf('%sx%i', modenames{n}, int32(hd));
                    end
                end
            end
        end
        
    end
    
    
end

pklist = [];
for n = 1:size(fatdot, 1); if ~isempty(modenames{n}); if numel(modenames{n}) <= 2; pklist(end+1) = n; end; end; end
nPeaks = numel(pklist);

% Identify 2nd order intermodulation
for n = 1:size(fatdot, 1)
    if isempty(modenames{n}) % not identified as a mode
        for u=1:nPeaks
            for v = (u+1):nPeaks
                if abs(fatdot(pklist(u),1) + fatdot(pklist(v),1) - fatdot(n,1)) < 1/tfunda
                    modenames{n} = [modenames{pklist(u)} '+' modenames{pklist(v)}];
                end
                
                if abs(fatdot(pklist(u),1) - fatdot(pklist(v),1) - fatdot(n,1)) < 1/tfunda
                    modenames{n} = [modenames{pklist(u)} '-' modenames{pklist(v)}];
                end
                
                if abs(fatdot(pklist(v),1) - fatdot(pklist(u),1) - fatdot(n,1)) < 1/tfunda
                    modenames{n} = [modenames{pklist(u)} '-' modenames{pklist(v)}];
                end
            end
        end
    end
        
end

for n = 1:size(fatdot,1)
    s0 = modenames{n};
    fi = fatdot(n,1);
    fprintf('Spectral peak at f = %f = %f f0 has magnitude %f', fi, fi / ffunda, fatdot(n,2));
    if numel(s0) > 0
        if numel(s0) > 2
            if s0(3) == 'x'
                fprintf(' - %s harmonic distortion\n', s0);
            else
                fprintf(' - %s intermodulation\n', s0);
            end
        else
            fprintf(' - %s peak identified\n', s0);
        end
    else
        fprintf('\n');
    end
end

legend('X_{shock} spectrum',  'Relative luminosity spectrum', 'Detected peaks');
    
hold off;
pwd
fprintf('Normalized Peak-peak luminosity fluctuation is %f.\n', lpp);

modelist = {'F','1O','2O','3O','4O','5O','6O','7O','8O','9O','10O'};
datablock = zeros([11 3]); % rows of [frequency xshock_amp radiance_amp] for mode F to 10

for q = 1:numel(pklist)
    qp = pklist(q);
    for k=1:numel(modelist)
        if strcmp(modenames{qp},modelist{k}) == 1
            datablock(k, :) = [fatdot(qp,1)/(2*pi*tHat), fatdot(qp,2), raddot(qp,2)];
        end
    end
    
end

autovars = double([pts(1) pts(2) pointspace nlpoint(1) endpt]);
save('autovars.mat','autovars');

if exist('reenableAuto','var')
    rhdAutodrive = 1;
    clear reenableAuto;
end

if exist('self', 'var') && isa(self, 'FMHandler2')
    disp('"self" exists: Assuming I am running inside FMHandler.autoanalyzeEntireDirectory on automatic.');
    if round(100*self.gamma) == runparams.gamma
        self.insertPointNew(runparams.m, runparams.theta, datablock);
    end
else
    if runparams.gamma == 167
        if exist('f53','var')
            f53.insertPointNew(runparams.m, runparams.theta, datablock);
        elseif exist('self', 'var')
            
        else
            disp('Access to FMHandler directly is required to insert data.\n');
        end
    elseif runparams.gamma == 140
        if exist('f75','var')
            f75.insertPointNew(runparams.m, runparams.theta, datablock);
        else
            disp('Access to FMHandler directly is required to insert data.\n');
        end
    elseif runparams.gamma == 129
        if exist('f97','var')
            f97.insertPointNew(runparams.m, runparams.theta, datablock);
        else
            disp('Access to FMHandler directly is required to insert data.\n');
        end
    else
        disp('Strange, an adiabatic index that is not 5/3, 7/5 or 9/7?\n');
    end
end

%end
