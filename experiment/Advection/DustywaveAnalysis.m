function autopsy = DustywaveAnalysis(directory, runParallel)

dir0 = pwd();
cd(directory);
load('SimInitializer_rank0.mat','IC');
cd(dir0);

S = SavefilePortal(directory);

% If called by Imogen, this will be true (such that the parallel process can run in parallel)
% If the savefile portal's parallel switch is flipped on, loaded frames are only this rank's
% frame segment instead of the reassembled entire frame.
if nargin < 2; runParallel = 0; end

S.setParallelMode(runParallel);
S.setFrametype(7); % Access 3D data frame by frame

% Pick out the wavevector
omega = IC.ini.waveOmega; 
Kvec = IC.ini.pWaveK;
%Kmag = norm(Kvec);
%Khat = Kvec / Kmag;
evec = IC.ini.waveEigenvector;

% Calculate the initial phases we'll use to project the result onto the
% full simulation grid
geo = GeometryManager(IC.ini.geometry.globalDomainRez);
geo.makeBoxSize(1); % FIXME HACK we should get this from the savefile portal
% not assume it
geo.makeBoxOriginCoord([-.5 -.5 -.5]);

[xv, yv, zv] = geo.ndgridSetIJK('pos');
KdotX = xv*Kvec(1) + yv*Kvec(2) + zv*Kvec(3);

% Calculate the linear factor in displacement
%machParallel = IC.ini.backgroundMach' * Khat;
%machPerp     = IC.ini.backgroundMach' - Khat' * machParallel;

% Output vars...
rhoerr_L1 = zeros([1 S.numFrames]);
rhoerr_L2 = zeros([1 S.numFrames]);
%velerr_L1 = []; velerr_L2 = [];
frameT = zeros([1 S.numFrames]);

amp = IC.ini.pAmplitude(1);

% Iterating over all frames in sequence,
for N = 1:S.numFrames
    F = S.nextFrame();
    % In actuality, our 'error' is asserting that the length of a wave is
    % 1. But we'd have to remap a whole grid of Xes, so we just scale time the opposite way    
    t = sum(F.time.history);
    
    rhogt = IC.ini.pDensity(1) + imag(amp*evec(1)*exp(1i*KdotX - 1i*omega*t));
    
    % The moment of truth: calculate the 1- and 2-norms
    delta = rhogt - F.mass;
    if runParallel; delta = geo.withoutHalo(delta); end

    frameT(N) = t;

    rhoerr_L1(N) =      mpi_sum(norm(delta(:),1)  ) / mpi_sum(numel(delta)) ;
    rhoerr_L2(N) = sqrt(mpi_sum(norm(delta(:),2)^2) / mpi_sum(numel(delta)));
end

autopsy.T = frameT;
autopsy.resolution = size(F.mass);
autopsy.wavenumber = IC.ini.pWavenumber;

autopsy.rhoL1 = rhoerr_L1;
autopsy.rhoL2 = rhoerr_L2;

%autopsy

if mpi_amirank0()
    cd(directory);
    save('analyzer_results.mat','autopsy');
    cd(dir0);
end

return


