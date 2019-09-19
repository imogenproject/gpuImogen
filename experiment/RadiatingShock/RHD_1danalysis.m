load 4D_XYZT

F = DataFrame(F);

x = trackFront2(squeeze(F.mass), (1:size(F.mass,1))*F.dGrid{1}, .5*(F.gamma+1)/(F.gamma-1));

% Automatically strip junk at the end of the run off if it hit the end of the grid
N = RHD_utils.lastValidFrame(F, x);

if N < size(F.mass,4)
    fprintf('Note: shock gets too close to end of grid at frame %i: Truncating dataset.\n', N);
    %F.truncate([], [], [], 1:N);
    %x=x(1:N);
end

hold off;
% throw up pretty spacetime diagram
imagesc(diff(squeeze(F.mass), 1, 2)./squeeze(F.mass(:,1,1,1:(end-1))));
thefig = gca();
thefig.CLim = [-.3 .3];

vx = squeeze(F.velX);
cs = squeeze(sqrt(F.gamma*F.pressure ./ F.mass));

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

% This block projects parabolas onto the left and right areas indicated
% and takes their nearby intercept as the true shock bounce point;
% This is much better than the nearest-guess method previously used.
cla('reset');

plot((pts(1)-10):(pts(2)+10), x((pts(1)-10):(pts(2)+10)));
hold on;

pointspace = input('Enter value to force pointspace or blank for default=2: ');
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

tfunda = interp1(1:size(F.time.time), F.time.time, [interframe1 interframe2],'linear');

plot([interframe1 interframe2], x(round([interframe1 interframe2])), 'kv', 'MarkerSize', 10);
hold off;

tfunda0 = F.time.time(pts);
ffunda0 = 1/(tfunda0(2) - tfunda0(1));

ffunda = 1/(tfunda(2)-tfunda(1));

dfreq = abs(ffunda/ffunda0 - 1);
if dfreq > .03
    disp(['WARNING: 0th and 2nd order fundamental frequencies differ by ' num2str(100*dfreq) '%... you sure those were bounce points chief?']); 
else
    disp(['2nd order extrapolation of ' num2str(ffunda) ' differs from 0th order estimate by ' num2str(100*dfreq) '%, good.']);
end

input('Enter if satisfied with projection: ');

plot(x(3:end));
nlpoint = [];
while isempty(nlpoint); nlpoint = input('Frame to start spectral analysis at? '); end

if numel(nlpoint) > 1
    endpt = nlpoint(2);
else
    endpt = numel(F.time.time);
end
    
spacing = round(pts(2)-pts(1));
cycles = round(endpt-nlpoint(1))/(interframe2 - interframe1) - 1;

endpt = RHD_utils.walkdown(x, round(nlpoint(1) + cycles*(interframe2-interframe1)), spacing);
stpt = RHD_utils.walkdown(x, nlpoint(1), spacing);

if 0
npc = nlpoints_class(x(1:endpt), F.time.time(1:endpt), nlpoint);

timepts = F.time.time(nlpoint:endpt);
pospts = x(nlpoint:endpt)';

[coeffs, resid] = polyfit(timepts, pospts, 1);
oscil = pospts - (coeffs(1)*timepts + coeffs(2));
plot(timepts, oscil);

dump = input('Use up/down arrow to get the endpoints to match & minimize spectral garbage, then press enter: ');
stpt = npc.leftpoint;
clear npc;
end

npt = 1+endpt-stpt;

if npt/2 ~= round(npt/2); stpt = stpt-1; end

timepts = F.time.time(stpt:endpt);
pospts = x(stpt:endpt)';

[coeffs, resid] = polyfit(timepts, pospts, 1);
fprintf('Shock fallback velocity = %f\n', coeffs(1));

oscil = pospts - (coeffs(1)*timepts + coeffs(2));

xfourier = 2*abs(fft(oscil))/numel(oscil);
xi = numel(xfourier(2:end/2));

tfunda = (timepts(end) - timepts(1));
% Generate harmonic markers
fquery = (1:16)*ffunda;

% This should pick up the F tone peak if it's an IF mode
fquery = [.6*ffunda fquery];

ypt = interp1((1:xi)/tfunda, xfourier(2:end/2), fquery, 'linear');

hold off;
plot((1:xi)/tfunda, xfourier(2:end/2),'b-');
hold on
plot(fquery, ypt, 'bo','markersize',7);

runparams = RHD_utils.parseDirectoryName(); % the worst way of learning this lol
rth = runparams.theta;

% Compute the luminosity on the given interval, normalized by L(t=0)
rr = RHD_utils.computeRelativeLuminosity(F, rth);

rft = 2*fft(rr(stpt:endpt)) / (1+endpt-stpt);

lpp = max(rr(stpt:endpt))-min(rr(stpt:endpt));

plot((1:xi)/tfunda,abs(rft(2:end/2)), 'r-');
ypt = interp1((1:xi)/tfunda, abs(rft(2:end/2)), fquery, 'linear');

plot(fquery, ypt, 'rx','markersize',7);

nneighbor = round(fquery*tfunda);

ispeak = @(y, xi) (y(xi) > y(xi+1)) & (y(xi) > y(xi-1)) & ( y(xi) > 2*sum(y(xi+[-2, -1, 1, 2])) );

isgausspeak = @(mag, center, std) (abs(center) < .55) & (std < 2);

grid on;

fatdot = [];
possibleFmode = 0;

for n = 1:16
    p = nneighbor(n)+1;
    if p < 5; continue; end
    if p - 4 > numel(xfourier); break; end
    
    % look up to 4 freq bins away
    p = RHD_utils.walkdown(xfourier, p, 4);
    
    [mag, center, std] = RHD_utils.gaussianPeakFit(xfourier, p);
    
    if isgausspeak(mag, center - p, std)
        if n == 1; possibleFmode = 1; end
        fatdot(end+1,:) = [center-1, mag, std];
    end
end

spec_residual = RHD_utils.subtractKnownPeaks(xfourier, fatdot);

if size(fatdot,1) > 0
    plot(fatdot(:,1)/tfunda, fatdot(:,2), 'kv', 'MarkerSize', 8);
    
    %[~, sortidx] = sort(fatdot(:,2),'descend');
    %fatdot = fatdot(sortidx, :);
    fprintf("Frequency resolution = +-%f\n", 1/tfunda);
    for n = 1:size(fatdot,1)
        fi = fatdot(n,1)/tfunda;
        fprintf('Spectral peak at f=%f = %f f0 has magnitude %f\n', fi, fi / ffunda, fatdot(n,2));

    end
    legend('X_{shock} spectrum',  'Harmonics of indicated period', 'Possible base tones', 'Relative luminosity spectrum');
    
    if possibleFmode
        disp('There appears to be a peak just below 1/3 of the 1O mode frequency: This may be an IF mode!'); 
        if abs(fatdot(2,1)/fatdot(1,1) - 3.3) < .25
            fb = 2;
        else
            fb = 3;
        end
        rat = fatdot(1,2)/fatdot(fb,2);
        
        if rat < 3
            fprintf('Mode amplitude ratio = %f < 3: Suggest this be classed as a 1O\n', rat);
        else
            fprintf('Mode amplitude ratio = %f > 3: Suggest this be classed as an F/IF\n', rat);
        end
    end
else
    disp('Well this is embarassing; No multiples of Ffunda hit a spectral peak...');
    legend('X_{shock} spectrum',  'Harmonics of indicated period', 'Relative luminosity spectrum');
end

hold off;
pwd
fprintf('Normalized Peak-peak luminosity fluctuation is %f.\n', lpp);

cleanspectrum = 1;
% spectral purity test
if size(fatdot,1) >= 3
    % Check ratio of 2nd and 3rd peaks: must be multiples of dominant tone
    mr = fatdot(2,1)/fatdot(1,1);
    if abs(mr - round(mr)) > .01; cleanspectrum = 0; end
    mf = fatdot(3,1)/fatdot(1,1);
    if abs(mr - round(mr)) > .01; cleanspectrum = 0; end
    
    if cleanspectrum
        resid = rms(spec_residual(2:(end/2)));
        tspec = rms(xfourier(2:(end/2)));
        if  resid > .15*tspec
            cleanspectrum = 0;
            fprintf('Low spectral purity: rms(residual) = %f x rms(spectrum) is > 15%%.\n', resid/tspec);
        else
            domF = fatdot(1,1)/tfunda;
            domMode = round(domF/ffunda)-1;
            fprintf('Clean spectrum detected: Automatically inserting m=%f, theta=%f, F=%f, m=%i\n', runparams.m, runparams.theta, domF, domMode);
        end
    else
        disp('Low spectral purity: Top 3 most powerful tones are not harmonic multiples of f0') 
    end
else
    cleanspectrum = 0;
    disp('Spectrum unclear: Less than 3 peaks aligned with f0 found');
end
    
if cleanspectrum == 0
    disp('Automatic data insertion off');
    fm = input('User override: Enter [frequency, mode] or just f manually, or empty to bypass: ');
    
    if numel(fm)==2
        domF = fm(1);
        domMode = round(fm(2));
        cleanspectrum = 1;
    elseif numel(fm)==1
        domF = fm;
        domMode = round(fm / ffunda)-1;
        cleanspectrum = 1;
    else
        
    end
end

if cleanspectrum
    if runparams.gamma == 167
        if exist('f53','var')
            f53.insertPoint(runparams.m, runparams.theta, domF, domMode);
            f53.insertLumi(runparams.m, runparams.theta, lpp);
        else
            fprintf('f53.insertPoint(%f, %f, %f, %f);\nf53.insertLumi(%f, %f, %f);\n', runparams.m, runparams.theta, domF, domMode, runparams.m, runparams.theta, lpp);
        end
    elseif runparams.gamma == 140
        if exist('f75','var')
            f75.insertPoint(runparams.m, runparams.theta, domF, domMode);
            f75.insertLumi(runparams.m, runparams.theta, lpp);
        else
            fprintf('f75.insertPoint(%f, %f, %f, %f);\nf75.insertLumi(%f, %f, %f);\n', runparams.m, runparams.theta, domF, domMode, runparams.m, runparams.theta, lpp);
        end
    elseif runparams.gamma == 129
        if exist('f97','var')
            f97.insertPoint(runparams.m, runparams.theta, domF, domMode);
            f97.insertLumi(runparams.m, runparams.theta, lpp);
        else
            fprintf('f97.insertPoint(%f, %f, %f, %f);\nf97.insertLumi(%f, %f, %f);\n', runparams.m, runparams.theta, domF, domMode, runparams.m, runparams.theta, lpp);
        end
    else
        
    end
end
