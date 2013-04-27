function waterfall = AdvectionAnimate(inputdir)

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
waterfall=[];

for N = 1:numel(filelist)
    f = util_LoadWholeFrame('3D_XYZ',5,filelist(N) );

    m = f.mass(:,1);
    dx = sqrt(5/3)*sum(f.time.history);

    m = circshift(m, -round(IC.ini.grid(1)*dx));
    waterfall(:,N) = m;
    fprintf('%i ', numel(filelist)-N);
end
fprintf('\n');


end
