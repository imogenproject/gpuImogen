function result = tsDustywave(iniResolution, doublings, kcouple, prettyPictures, methodPicker)
%doublings = 7; % will run from 32x1 to 2Kx1
%w0 = 1.5;
% The double blast wave test is a classic test for the shock-capturing codes
% testing the ability to handle very strong (M >> 1) shocks, as well
% as colliding shocks

if nargin < 4
    prettyPictures = 0;
end

grid = iniResolution;
%--- Initialize test ---%
run             = AdvectionInitializer(grid);
% run.timeMax     = 0.038; % set by run.cycles
run.iterMax     = 50000;

run.alias       = '';
run.info        = '1D Double Blast Wave test.';
run.notes       = '';

run.ppSave.dim3 = 25;
run.cfl = 0.45;

% Set a background speed at which the fluid is advected
run.backgroundMach = -0;

% for 2 fluids, sonic is the coupled propagating mode
% the evanescent mode is 'dustydamp.'
run.waveType = 'sonic';
run.amplitude = .0001;

% FWIW an amplitude of .0001 corresponds to a roughly 100dB sound in air
%                      .01                    roughly 140dB

% number of transverse wave periods in Y and Z directions
run.wavenumber = [1 0 0];
% 1st method of setting run duration: normalized by cycle time
run.cycles = 32;

run.addNewFluid(1);

run.fluidDetails(1) = fluidDetailModel('cold_molecular_hydrogen');
run.fluidDetails(2) = fluidDetailModel('10um_iron_balls');
run.fluidDetails(2).sigma = run.fluidDetails(2).sigma * 1e-3 * kcouple^2;
% rescale this to set Kcouple
% FIXME this will not handle nonlinear drag correctly for obvious reasons!!!
run.fluidDetails(2).mu = run.fluidDetails(2).mu * 1e-5 * (.495 / 1 ) * 8 / norm(run.wavenumber);

run.writeFluid = 2;
  run.amplitude = 0;
  run.backgroundMach = 0;
  run.setBackground(1, .001);

run.alias= 'dustybox';

run.ppSave.dim3 = 100;
  
fm = FlipMethod(); % 1 = HLL, 2 = HLLC, 3 = XJ
  fm.iniMethod = 2; 
%  fm.toMethod = 2;
%  fm.atstep = -1;
run.peripherals{end+1} = fm;


if prettyPictures
    rp = RealtimePlotter();
  rp.plotmode = 5;
  rp.plotDifference = 0;
  rp.insertPause = 1;
  rp.forceRedraw = 1;
  rp.iterationsPerCall = 1;
  rp.firstCallIteration = 1;
  rp.firstCallTime = 1;
  rp.spawnGUI = 1;
  
  rp.plotmode = 4;
rp.cut = [round(grid(1)/2), 1, 1];
rp.indSubs = [1 1 grid(1);1 1 1;1 1 1];
rp.movieProps(0, 0, 'RTP_');
rp.vectorToPlotprops(1, [1   1   0   1   1   1   0   1   0   1  10   1   8   1]);
rp.vectorToPlotprops(2, [1   5   0   1   1   1   0   1   0   1  10   1   8   1]);
rp.vectorToPlotprops(3, [2   1   0   1   1   1   0   1   0   1  10   1   8   1]);
rp.vectorToPlotprops(4, [2   5   0   1   1   1   0   1   0   1  10   1   8   1]);

%run.peripherals{end+1} = rp;

end
if nargin == 5
    run.peripherals{end+1} = methodPicker;
end

run.waveLinearity(0);
run.waveStationarity(0);

run.info        = '';
run.notes       = '';

run.image.parallelUniformColors = true;

result.N = [];
result.L1 = [];
result.L2 = [];
result.paths={};

outdirs = cell(doublings,1);

for N = 1:doublings
    % Run tests
    disp(['Running at resolution: ',mat2str(grid)]);
    run.geomgr.setup(grid);
    icfile   = run.saveInitialCondsToFile();
    outdirs{N}   = imogen(icfile);
    
    grid(1) = grid(1)*2;
    if prettyPictures
        rp.cut = [round(grid(1)/2), 1, 1];
        rp.indSubs = [1 1 grid(1);1 1 1;1 1 1];
    end
end

rhoGs = cell(doublings, 1);
rhoDs = cell(doublings, 1);
phases =cell(doublings, 1);

for N = 1:doublings;
    enforceConsistentView(outdirs{N});
    S = SavefilePortal(outdirs{N});
    S.setFrametype('X');
    
    F = S.jumpToLastFrame();

    rhoGs{N} = F.mass;
    rhoDs{N} = F.mass2;
    
    ic = S.getInitializerFile();
    phases{N} = ((1:size(F.mass,1))'+0.5)*2*pi*ic.ini.wavenumber(1) / size(F.mass,1);
    
     ev = conj(ic.ini.waveEigenvector);
     w0 = -conj(ic.ini.waveOmega);
     tau = sum(F.time.history);
        
    if prettyPictures
        figure(1);

        hold off;
        % gas rho
        plot((F.mass - ic.ini.pDensity(1) - imag(ic.ini.amplitude(1) * ev(1) * exp(1i*(phases{N} - w0*tau))))/ic.ini.amplitude(1),'r-o'); 
        hold on;
        plot((F.mass2 - ic.ini.pDensity(2) - imag(ic.ini.amplitude(1) * ev(4) * exp(1i*(phases{N} - w0*tau))))/ic.ini.amplitude(1),'r-x');
        % gas V
        plot((F.momX ./ F.mass - imag(ic.ini.amplitude(1) * ev(2) * exp(1i*(phases{N} - w0*tau))))/ic.ini.amplitude(1),'g-o');
        % dust v
        plot((F.momX2 ./ F.mass2 - imag(ic.ini.amplitude(1) * ev(5) * exp(1i*(phases{N} - w0*tau))))/ic.ini.amplitude(1),'g-x');

        title(['Resolution: ' num2str(size(F.mass,1)) ]);
        
        
        figure(2);

        hold off;
        % gas rho
        plot((F.mass - ic.ini.pDensity(1))/ic.ini.amplitude(1),'ro'); 
        hold on;
        plot(( imag(ic.ini.amplitude(1) * ev(1) * exp(1i*(phases{N} - w0*tau))))/ic.ini.amplitude(1),'r-'); 

        plot((F.mass2 - ic.ini.pDensity(2))/ic.ini.amplitude(1),'rx');
        plot((imag(ic.ini.amplitude(1) * ev(4) * exp(1i*(phases{N} - w0*tau))))/ic.ini.amplitude(1),'r-');
        % gas V
        plot(F.momX ./ F.mass/ic.ini.amplitude(1) ,'go');
        plot((imag(ic.ini.amplitude(1) * ev(2) * exp(1i*(phases{N} - w0*tau))))/ic.ini.amplitude(1),'g-');
        % dust v
        plot(F.momX2 ./ F.mass2/ic.ini.amplitude(1),'gx');
        plot((imag(ic.ini.amplitude(1) * ev(5) * exp(1i*(phases{N} - w0*tau))))/ic.ini.amplitude(1),'g-');

        title(['Resolution: ' num2str(size(F.mass,1)) ]);
        
        dumpme = input('Enter to continue: ');
    end
end

% We wish to weigh the gas and the dust equally.
rat = mean(F.mass2(:)) / mean(F.mass(:));

rhoGbar = rhoGs{N};
rhoDbar = rhoDs{N};

if doublings > 1
    for N = (doublings-1):-1:1
        % average 2:1
        rhoGbar = (rhoGbar(1:2:end) + rhoGbar(2:2:end))/2;
        rhoDbar = (rhoDbar(1:2:end) + rhoDbar(2:2:end))/2;
        
        result.N(N)  = numel(rhoGbar);
        ng = norm((rhoGs{N}-rhoGbar)/numel(rhoGbar), 1);
        nd = norm((rhoDs{N}-rhoDbar)/numel(rhoDbar), 1);
        result.L1(N) =      mpi_sum(norm([ng(:); rat*nd(:)],1)  ) / mpi_sum(numel(ng)) ;
        result.L2(N) = sqrt(mpi_sum(norm([ng(:); rat*nd(:)],2)^2) / mpi_sum(numel(ng)));
        
        lin_drho_g = rhoGs{N} - (1 + imag(ic.ini.amplitude(1) *ev(1) * exp(1i*(phases{N} - w0*tau))));
        lin_drho_d = rhoDs{N} - (1 + imag(ic.ini.amplitude(1) *ev(4) * exp(1i*(phases{N} - w0*tau))));
        
        result.L1_linear(N) = mpi_sum(norm([lin_drho_g(:); rat*lin_drho_d(:)],1)  ) / mpi_sum(numel(ng)) ;
        result.L2_linear(N) = sqrt(mpi_sum(norm([lin_drho_g(:); rat*lin_drho_d(:)],2)^2) / mpi_sum(numel(ng)));
    end
else
    result.N  = numel(rhoGbar);
    lin_drho_g = rhoGs{N} - (1 + imag(ic.ini.amplitude(1) *ev(1) * exp(1i*(phases{N} - w0*tau))));
    lin_drho_d = rhoDs{N} - (1 + imag(ic.ini.amplitude(1) *ev(4) * exp(1i*(phases{N} - w0*tau))));
        
    result.L1_linear(N) = mpi_sum(norm([lin_drho_g(:); rat*lin_drho_d(:)],1)  ) / mpi_sum(result.N ) ;
    result.L2_linear(N) = sqrt(mpi_sum(norm([lin_drho_g(:); rat*lin_drho_d(:)],2)^2) / mpi_sum(result.N ));
end

result.paths = outdirs;

if doublings > 1
    figure(3);
    
    if prettyPictures
        hold off
        plot(log2(result.N), log2(result.L1),'b-')
        hold on
        plot(log2(result.N), log2(result.L2),'r')
        plot(log2(result.N), log2(result.L2_linear),'rx')
        plot(log2(result.N), log2(result.L1_linear),'bx')
    end
end

if mpi_amirank0()
    d0 = pwd();
    cd(outdirs{1});
%    save('./tsCentrifugeResult.mat','result');
    cd(d0);
end

end
