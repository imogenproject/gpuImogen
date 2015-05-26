function autopsy = AdvectionAnalysis(directory)

dir0 = pwd();
cd(directory);
load('SimInitializer_rank0.mat','IC');
cd(dir0);

S = SavefilePortal(directory);
S.setFrametype(7); % Access 3D data frame by frame

% Pick out the wavevector
Kvec = IC.ini.pWaveK;
Kmag = norm(Kvec);
Khat = Kvec / Kmag;

% Compute displacments of a fine grid of characteristic amplitudes
erange = IC.ini.amplitude*(-1.0001:.0001:1.0001)*IC.ini.pDensity;

c0 = sqrt(IC.ini.gamma * IC.ini.pPressure / IC.ini.pDensity);

% Calculate the initial phases we'll use to project the result onto the
% full simulation grid
G = GlobalIndexSemantics();
G.setup(IC.ini.grid);
[xv yv zv] = G.ndgridSetXYZ([1 1 1], IC.ini.dGrid);
KdotX = xv*Kvec(1) + yv*Kvec(2) + zv*Kvec(3);

% Calculate the linear factor in displacement
machParallel = IC.ini.backgroundMach * Khat';
machPerp     = IC.ini.backgroundMach - Khat * machParallel;

% Output vars...
rhoerr_L1 = []; rhoerr_L2 = [];
velerr_L1 = []; velerr_L2 = [];
frameT = [];

% Since we know the initial function is a sine, write down the critical
% time
tCritical = 2/((IC.ini.gamma + 1)*c0*Kmag*IC.ini.amplitude);

% Iterating over all frames in sequence,
for N = 1:S.numFrames();
    F = S.nextFrame();
    t = sum(F.time.history);
    % Compute the displacement of a reference wave through circular BCs 
    % Parameterized by original phase
    origMap = CharacteristicAnalysis1D(0:.0001:.9999, 1, IC.ini.pDensity, c0, machParallel, IC.ini.gamma, IC.ini.amplitude*cos(2*pi*(0:.0001:.9999)), t);

    % Map this onto the full 3D space by referring to original phases
    rhoAnalytic = interp1(2*pi*(0:.0001:.9999), origMap, mod(KdotX,2*pi),'cubic');

    % The moment of truth
    delta = (rhoAnalytic - F.mass);
    if sum(F.time.history) >= tCritical;
        disp(['At frame', S.tellFrame(), ' time ', num2str(t), ' exceeded tCritical=', num2str(tCritical),'; Analysis ended.'])
        break;
    end
    
    frameT(end+1) = t;

    rhoerr_L1(N) = norm(delta,1) / numel(delta);
    rhoerr_L2(N) = norm(delta,2) / sqrt(numel(delta));
end

autopsy.T = frameT;
autopsy.resolution = size(F.mass);
autopsy.wavenumber = IC.ini.pWavenumber;

autopsy.rhoL1 = rhoerr_L1;
autopsy.rhoL2 = rhoerr_L2;

save('analyzer_results.mat','autopsy');

return


