function [result errors] = AdvectionAnalyzer(inputdir)

% 1. Go to input directory, load rank 0 initializer.
if nargin == 0; inputdir = input('No directory given. Path to advection test? ', 's'); end
cd(inputdir);

% 2. Get box wavenumbers
load('SimInitializer_rank0.mat');

if IC.ini.grid(3) == 1; DIMENSION = 2; else; DIMENSION = 3; end

waveNumber = IC.ini.waveK;
waveVector = IC.ini.waveDirection*2*pi*waveNumber ./ (IC.ini.dGrid.*IC.ini.grid); % Calculate realspace wavevectors = 2 pi N_i / L_i
A0 = IC.ini.waveAmplitude;

omega = IC.ini.waveDirection*sqrt(5/3)*norm(waveVector);

% 3. Get from user savefile frame IDs
!ls | grep XYZ_rank0 | sed -e 's/_/ /g' -e 's/\./ /' | awk '/.*/ { printf("%s\n",$4); }' | awk '/^[0-9]/ { printf("%s ",$1); }' > tempnumbersfile
filelist = dlmread('tempnumbersfile');
filelist = [0 filelist];
!rm tempnumbersfile

% 4. Compare amplitude/phase at savefile times with known exact solutions
result = [];
for N = 1:numel(filelist)
    f = util_LoadWholeFrame('3D_XYZ',5,filelist(N) );

    dq = fftn(f.mass);
    fc = dq(waveNumber(1)+1, waveNumber(2)+1, waveNumber(3)+1);
    result(N,:) = [sum(f.time.history) abs(fc) angle(fc)  ]; 
end

result(:,3) = unwrap(result(:,3)); % unwrap phase
result(:,4) = result(1,3)-omega*result(:,1); % Subtract predicted phase;
% 5. store this and possibly plot it.

gamma  = polyfit(result(:,1), log(result(:,2)),1);
phidot = polyfit(result(:,1), result(:,3)-result(:,4), 1);

errors(1) = gamma(1);
errors(2) = phidot(1);

fprintf('error omega w_er in exp(-i w_er t) is %e%ei\n', -phidot(1), gamma(1));

end
