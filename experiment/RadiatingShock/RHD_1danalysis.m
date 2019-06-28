load 4D_XYZT

F = DataFrame(F);

x = trackFront2(squeeze(F.mass), (1:size(F.mass,1))*F.dGrid{1}, 2.5);

imagesc(diff(squeeze(F.mass), 1, 2)./squeeze(F.mass(:,1,1,1:(end-1))));
thefig = gca();
thefig.CLim = [-.3 .3];

pts = input('Identify frames demarking one round-trip: ');

tfunda = F.time.tFrame(pts);
ffunda = 1/(tfunda(2)-tfunda(1))

plot(x);

nlpoint = input('Saturation of nonlinearity frame? ');

ttotal = cumsum(F.time.history);
ttotal = [0; ttotal(1:800:end)]; % hack: assumption!

timepts = F.time.tFrame(nlpoint:end);
pospts = x(nlpoint:end)';

[coeffs, resid] = polyfit(timepts, pospts, 1)

oscil = pospts - (coeffs(1)*timepts + coeffs(2));

plot(timepts, oscil)

dump = input('Enter to continue if satisfactorily flat residual: ');

xf = log(abs(fft(oscil)));
plot(xf(2:end/2));

xi = numel(xf(2:end/2));

tfunda = timepts(end) - timepts(1);

plot((1:xi)/tfunda, xf(2:end/2));

peaks = input('Input list of spectral peaks: ');

disp('F/F_round trip: ')
disp(peaks / ffunda)

dump = input('enter to continue');

ntime = size(F.mass,4);

for N = (ntime-25):ntime
	plot(log(squeeze(F.mass(:,1,1,N)))); title(N);
	dump = input('enter to cont');
end

