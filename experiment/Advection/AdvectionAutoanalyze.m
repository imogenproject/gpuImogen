function domega = AdvectionAnalyzer(inputdir)

cd(inputdir);
% 2. Get box wavenumbers
load('SimInitializer_rank0.mat');

if IC.ini.grid(3) == 1; DIMENSION = 2; else; DIMENSION = 3; end

waveNumber = IC.ini.waveK;
waveVector = IC.ini.waveDirection*2*pi*waveNumber ./ (IC.ini.dGrid.*IC.ini.grid); % Calculate realspace wavevectors = 2 pi N_i / L_i
A0 = IC.ini.waveAmplitude;

% omega = c_s K; ASSUMES SONIC WAVE
omega = IC.ini.waveDirection*sqrt(5/3)*norm(waveVector);

% 3. Get from user savefile frame IDs
%!ls | grep XYZ_rank0 | sed -e 's/_/ /g' -e 's/\./ /' | awk '/.*/ { printf("%s\n",$4); }' | awk '/^[0-9]/ { printf("%s ",$1); }' > tempnumbersfile
%filelist = dlmread('tempnumbersfile');
%filelist = [0 filelist];
%!rm tempnumbersfile

% 4. Compare amplitude/phase at savefile times with known exact solutions
result = [];
f0 = util_LoadWholeFrame('3D_XYZ',5,0);
f = util_LoadWholeFrame('3D_XYZ',5,9999);

dq = fftn(f.mass);
fc = 2*dq(waveNumber(1)+1, waveNumber(2)+1, waveNumber(3)+1) / numel(f.mass);

q0 = fftn(f0.mass);
fc0= 2*q0(waveNumber(1)+1, waveNumber(2)+1, waveNumber(3)+1) / numel(f.mass);

% Solve A1 = A0 exp(w_im t) with w_im the error in amplitude for w_re
omega_im = log(abs(fc)/A0) / sum(f.time.history); 


% Solve phi(t) = (w0 + w_re) t for w_re the error in phase
omega_re = -(angle(fc) - angle(fc0))/sum(f.time.history);
%omega_re = (angle(fc) - angle(fc0)) / (sum(f.time.history)*IC.ini.numWavePeriods);

fprintf('Wave transport analysis:\nTheory: q = q0 exp(-i w0 t)\n');
fprintf('	c_s = %12.4e\n	k = 2 pi [%i %i %i]\n',sqrt(5/3), waveNumber(1),waveNumber(2),waveNumber(3));
fprintf('        w0 = %12.4e\n',omega);
fprintf('Reality: q = q0 exp(-i(w0 + dw)t)\n	t = %12.4e\n	Re[dw] = %12.4e,\n	Im[dw] = %12.4e\n',sum(f.time.history),omega_re, omega_im);
fprintf('Normalized by w0: dw_hat = %12.4e %12.4ei\n', omega_re/omega, omega_im/omega);
%fprintf('error omega w_err in exp(-i w_err t) is %12.4e %12.4ei\n', omega_im, omega_re);

domega = [ omega_re/omega, omega_im/omega];

end
