load 4D_XYZT

F = DataFrame(F);

x = trackFront2(squeeze(F.mass), (1:size(F.mass,1))*F.dGrid{1}, .5*(F.gamma+1)/(F.gamma-1));

hold off;
imagesc(diff(squeeze(F.mass), 1, 2)./squeeze(F.mass(:,1,1,1:(end-1))));
thefig = gca();
thefig.CLim = [-.3 .3];
drawnow;
hold off;
drawnow;

vx = squeeze(F.velX);
cs = squeeze(sqrt(F.gamma*F.pressure ./ F.mass));



pts = input('Identify frames demarking one round-trip: ');

pts(1) = RHD_utils.walkdown(x, pts(1), 5);
pts(2) = RHD_utils.walkdown(x, pts(2), 5);

% This block projects parabolas onto the left and right areas indicated
% and takes their nearby intercept as the true shock bounce point;
% This is much better than the nearest-guess method previously used.

plot((pts(1)-10):(pts(2)+10), x((pts(1)-10):(pts(2)+10)));
hold on;

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
nlpoint = input('Frame to start spectral analysis at? ');

if numel(nlpoint) > 1
    endpt = nlpoint(2);
else
    endpt = numel(F.time.time);
end
    

npc = nlpoints_class(x(1:endpt), F.time.time(1:endpt), nlpoint);

timepts = F.time.time(nlpoint:endpt);
pospts = x(nlpoint:endpt)';

[coeffs, resid] = polyfit(timepts, pospts, 1);
oscil = pospts - (coeffs(1)*timepts + coeffs(2));
plot(timepts, oscil);

dump = input('Use up/down arrow to get the endpoints to match & minimize spectral garbage, then press enter: ');
stpt = npc.leftpoint;
clear npc;

timepts = F.time.time(stpt:endpt);
pospts = x(stpt:endpt)';

[coeffs, resid] = polyfit(timepts, pospts, 1)

oscil = pospts - (coeffs(1)*timepts + coeffs(2));

xfourier = log(abs(fft(oscil)));
xi = numel(xfourier(2:end/2));

tfunda = (timepts(end) - timepts(1));
% Generate harmonic markers
fquery = (1:16)*ffunda;
ypt = interp1((1:xi)/tfunda, xfourier(2:end/2), fquery, 'linear');

plot((1:xi)/tfunda, exp(xfourier(2:end/2)));
hold on

rth = input('Radiative theta? ');
rr = RHD_utils.computeRelativeLuminosity(F, rth);

rft = fft(rr(stpt:endpt));

plot((1:xi)/tfunda,abs(rft(2:end/2)));

plot(fquery, exp(ypt), 'rx');

nneighbor = round(fquery*tfunda);

ispeak = @(y, xi) (y(xi) > y(xi+1)) & (y(xi) > y(xi-1)) & (y(xi-1) > y(xi-2)) & (y(xi+1) > y(xi+2));

fatdot = [];
for n = 1:16
    p = nneighbor(n)+1;
    if p - 4 > numel(xfourier); break; end
    
    % look up to 4 freq bins away
    p = RHD_utils.walkdown(xfourier, p, 4); 
    
    if ispeak(xfourier, p)
        fatdot(end+1,:) = [p-1, xfourier(p)];
    end
end
if size(fatdot,1) > 0
    plot(fatdot(:,1)/tfunda, exp(fatdot(:,2)), 'kv', 'MarkerSize', 8);
    
    [~, sortidx] = sort(fatdot(:,2),'descend');
    fprintf("Frequency resolution = +-%f\n", 1/tfunda);
    for n = 1:size(fatdot)
        fi = fatdot(sortidx(n),1)/tfunda;
       fprintf('Spectral peak at f=%f = %f f0 has magnitude %f\n', fi, fi / ffunda, exp( fatdot(sortidx(n),2)));
    end
    legend('X_{shock} spectrum', 'Relative luminosity spectrum', 'Harmonics of indicated period', 'Possible base tones');
else
    disp('Well this is embarassing; No multiples of Ffunda hit a spectral peak...');
    legend('Measured spectrum', 'harmonics of user-input period');
end

hold off;

peaks = input('Input list of other spectral peaks of interest if desired: ');

pwd
disp('F/F_round trip: ')
disp(peaks / ffunda)
disp(ffunda)






