function autopsy = SonicAdvectionAnalysis(directory, runParallel)

S = SavefilePortal(directory);

IC = S.getInitialConditions();

% If called by Imogen, this will be true (such that the parallel process can run in parallel)
% If the savefile portal's parallel switch is flipped on, loaded frames are only this rank's
% frame segment instead of the reassembled entire frame.
if nargin < 2; runParallel = 0; end

S.setParallelMode(runParallel);
S.setFrametype(7); % Access 3D data frame by frame

% Pick out the wavevector
Kvec = IC.ini.pWaveK;
Kmag = norm(Kvec);
Khat = Kvec / Kmag;

c0 = sqrt(IC.ini.gamma * IC.ini.pPressure / IC.ini.pDensity);

% Calculate the initial phases we'll use to project the result onto the
% full simulation grid
%geo = GeometryManager(IC.ini.geometry.globalDomainRez, [1 1 1]);
geo = GeometryManager(IC.ini.geometry);

%geo.makeBoxSize(1); % fixme hack we should get this from the savefile portal
% not assume it
%geo.makeBoxOriginCoord([-.5 -.5 -.5]);

[xv, yv, zv] = geo.ndgridSetIJK('pos');
KdotX = xv*Kvec(1) + yv*Kvec(2) + zv*Kvec(3);

% Calculate the linear factor in displacement
machParallel = IC.ini.backgroundMach' * Khat;
%machPerp     = IC.ini.backgroundMach' - Khat' * machParallel;

% Output vars...[]
qq = zeros([S.numFrames 1]);
rhoerr_L1 = qq; rhoerr_L2 = qq;
%velerr_L1 = qq; velerr_L2 = qq;
frameT = qq;

% Since we know the initial function is a sine, write down the critical
% time
tCritical = 2/((IC.ini.gamma + 1)*c0*Kmag*IC.ini.amplitude);

% Iterating over all frames in sequence,
for N = 1:S.numFrames
    F = S.nextFrame();
    % In actuality, our 'error' is asserting that the length of a wave is
    % 1. But we'd have to remap a whole grid of Xes, so we just scale time the opposite way    
    t = F.time.time * norm(IC.ini.pWavenumber);
    
    % Compute the displacement of a reference characteristic packet
    % Parameterized by original phase
    rng = 0:.0001:.9999;
    if strcmp(IC.ini.waveType, 'sonic')
        backMap = CharacteristicAnalysis1D(rng, 1, IC.ini.pDensity, c0, machParallel, IC.ini.gamma, IC.ini.amplitude*cos(2*pi*rng), t);
    elseif strcmp(IC.ini.waveType, 'entropy')
        backMap = CharacteristicAnalysis1D_entropy(rng, 1, IC.ini.pDensity, c0, machParallel, IC.ini.gamma, IC.ini.amplitude*cos(2*pi*rng), t);
    end

    % Map this onto the full 3D space by referring to original phases
    rhoAnalytic = interp1(2*pi*rng, backMap, mod(KdotX,2*pi),'pchip');

    % The moment of truth: calculate the 1- and 2-norms
    delta = rhoAnalytic - F.mass;
    if runParallel; delta = geo.withoutHalo(delta); end

    if F.time.time >= tCritical
        disp(['At frame', num2str(S.tellFrame()), ' time ', num2str(t), ' exceeded tCritical=', num2str(tCritical),'; Analysis ended.'])
        break;
    end
    
    frameT(N) = t;

    rhoerr_L1(N) =      mpi_sum(norm(delta(:),1)  ) / mpi_sum(numel(delta)) ;
    rhoerr_L2(N) = sqrt(mpi_sum(norm(delta(:),2)^2) / mpi_sum(numel(delta)));
end

% Normalize by the L1/L2 measurement of the original perturbation
rhoerr_L1 = rhoerr_L1 / (4*IC.ini.amplitude);
rhoerr_L2 = rhoerr_L2 / (sqrt(pi)*IC.ini.amplitude);

autopsy.T = frameT;
autopsy.resolution = size(F.mass);
autopsy.wavenumber = IC.ini.pWavenumber;

autopsy.rhoL1 = rhoerr_L1;
autopsy.rhoL2 = rhoerr_L2;

%autopsy

if mpi_amirank0()
    cd(directory);
    dir0 = pwd();
    save('analyzer_results.mat','autopsy');
    cd(dir0);
end

return


