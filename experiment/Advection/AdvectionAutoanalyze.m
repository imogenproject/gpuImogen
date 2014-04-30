function dw = AdvectionAutoanalyze(inputdir)
% Given the directory storing an advection run,
% automatically loads it up and confirms accuracy.

cd(inputdir);
% 2. Get box wavenumbers
load('SimInitializer_rank0.mat');

if IC.ini.grid(3) == 1; DIMENSION = 2; else; DIMENSION = 3; end

waveNumber = IC.ini.waveN;
waveVector = IC.ini.waveK; % Calculate realspace wavevectors = 2 pi N_i / L_i
A0 = IC.ini.waveAmplitude;

omega = -IC.ini.waveOmega;

% 3. Get from user savefile frame IDs
%!ls | grep XYZ_rank0 | sed -e 's/_/ /g' -e 's/\./ /' | \
%awk '/.*/ { printf("%s\n",$4); }' | awk '/^[0-9]/ { printf("%s ",$1); }' > tempnumbersfile
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

t = sum(f.time.history);

% FIXME: The phase error calculation assumes the error is small (|| < pi)
% Solve A1 = A0 exp(w_im t) with w_im the error in amplitude for w_re
omega_im = log(abs(fc/fc0)) / t;
% Solve phi(t) = (w0 + w_re) t for w_re the error in phase
omega_re = -(angle(fc) - angle(fc0))/t;

dw = [ omega_re, omega_im] / abs(omega);

fprintf('Wave transport analysis:\nTheory: q = q0 exp(-i w0 t) = q(t)\n');
fprintf('\tc_s = %12.6e\n	k = 2 pi [%i %i %i]\n',sqrt(5/3), waveNumber(1),waveNumber(2),waveNumber(3));
fprintf('\tw0 = %12.6e\n',omega);
fprintf('\tt = %12.6e\n', sum(f.time.history));
fprintf('Reality: q = q(t) exp(-i w dw t)\n');
fprintf('\tRe[dw] / |w0| = %12.6e,\n\tIm[dw] / |w0| = %12.6e\n', dw(1), dw(2));
fprintf('\tphase error radians/cycle = %12.6e,\n\tdecay e-folds/cycle = %12.6e\n', abs(omega*t*dw(1))/IC.ini.numWavePeriods, abs(dw(2))/IC.ini.numWavePeriods);

end
