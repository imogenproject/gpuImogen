load 4D_XYZT

F = DataFrame(F);

x = trackFront2(squeeze(F.mass), (1:size(F.mass,1))*F.dGrid{1}, 2.5);

imagesc(diff(squeeze(F.mass), 1, 2)./squeeze(F.mass(:,1,1,1:(end-1))));
thefig = gca();
thefig.CLim = [-.3 .3];

pts = input('Identify frames demarking one round-trip: ');

% This block projects parabolas onto the left and right areas indicated
% and takes their nearby intercept as the true shock bounce point;
% This is much better than the nearest-guess method previously used.

% NOTE: Assumes at least 8 frames between shock bounces!!!
xleft = x(pts(1) - [7 4 1]);
xright= x(pts(1) + [1 4 7]);
% Convert to polynomials
xlpoly = polyfit( - [7 4 1], xleft, 2);
xrpoly = polyfit( + [1 4 7], xright,2);
% Solve intersection point: Assume it is the small-x one
deltapoly = xlpoly - xrpoly;

interframe1 = pts(1)+min(roots(deltapoly(end:-1:1)));

% NOTE: Assumes at least 8 frames between shock bounces!!!
xleft = x(pts(2) - [7 4 1]);
xright= x(pts(2) + [1 4 7]);
% Convert to polynomials
xlpoly = polyfit( - [7 4 1], xleft, 2);
xrpoly = polyfit( + [1 4 7], xright,2);
% Solve intersection point: Assume it is the small-x one
deltapoly = xlpoly - xrpoly;

interframe2 = pts(2)+min(roots(deltapoly(end:-1:1)));

tfunda = interp1(1:size(F.time.time), F.time.time, [interframe1 interframe2],'linear');

tfunda0 = F.time.time(pts);
ffunda0 = 1/(tfunda0(2) - tfunda0(1));

ffunda = 1/(tfunda(2)-tfunda(1));

dfreq = abs(ffunda/ffunda0 - 1);
if dfreq > .03
    disp(['WARNING: 0th and 2nd order fundamental frequencies differ by ' num2str(100*dfreq) '%... you sure those were bounce points chief?']); 
else
    disp(['2nd order extrapolation of ' num2str(ffunda) ' differs from 0th order estimate by ' num2str(100*dfreq) '%, good.']);
end

plot(x);

nlpoint = input('Frame to start spectral analysis at? ');

npc = nlpoints_class(x, F.time.time, nlpoint);

timepts = F.time.time(nlpoint:end);
pospts = x(nlpoint:end)';

[coeffs, resid] = polyfit(timepts, pospts, 1);
oscil = pospts - (coeffs(1)*timepts + coeffs(2));
plot(timepts, oscil);

dump = input('Use up/down arrow to get the endpoints to match & minimize spectral garbage, then press enter: ');

timepts = F.time.time(npc.leftpoint:end);
pospts = x(npc.leftpoint:end)';

[coeffs, resid] = polyfit(timepts, pospts, 1)

oscil = pospts - (coeffs(1)*timepts + coeffs(2));

xfourier = log(abs(fft(oscil)));
xi = numel(xfourier(2:end/2));

tfunda = (timepts(end) - timepts(1));
% Generate harmonic markers
fquery = (1:16)*ffunda;
ypt = interp1((1:xi)/tfunda, xfourier(2:end/2), fquery, 'linear');

plot((1:xi)/tfunda, xfourier(2:end/2));
hold on
plot(fquery, ypt, 'rx');

nneighbor = round(fquery*tfunda);

ispeak = @(y, xi) (y(xi) > y(xi+1)) & (y(xi) > y(xi-1)) & (y(xi-1) > y(xi-2)) & (y(xi+1) > y(xi+2));

fatdot = [];
for n = 1:16
    p = nneighbor(n)+1;
    if ispeak(xfourier, p)
        fatdot(end+1,:) = [p-1, xfourier(p)];
    elseif ispeak(xfourier, p-1)
        fatdot(end+1,:) = [p-2, xfourier(p-1)];
    elseif ispeak(xfourier, p+1)
        fatdot(end+1,:) = [p+0, xfourier(p+1)];
    end
end
if size(fatdot,1) > 0
    plot(fatdot(:,1)/tfunda, fatdot(:,2), 'kv', 'MarkerSize', 8);
    
    [~, sortidx] = sort(fatdot(:,2),'descend');
    fprintf("Frequency resolution = +-%f\n", 1/tfunda);
    for n = 1:size(fatdot)
        fi = fatdot(sortidx(n),1)/tfunda;
       fprintf('Spectral peak at f=%f = %f f0 has magnitude %f\n', fi, fi / ffunda, exp( fatdot(sortidx(n),2)));
    end
    legend('Spectral measurement', 'Harmonics of indicated period', 'Possible base tones');
else
    disp('Well this is embarassing; No multiples of Ffunda hit a spectral peak...');
    legend('Measured spectrum', 'harmonics of user-input period');
end

hold off;



peaks = input('Input list of spectral peaks: ');

disp('F/F_round trip: ')
disp(peaks / ffunda)

dump = input('enter to continue');

ntime = size(F.mass,4);

for N = (ntime-25):ntime
	plot(log(squeeze(F.mass(:,1,1,N)))); title(N);
	dump = input('enter to cont');
end



