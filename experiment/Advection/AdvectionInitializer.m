classdef AdvectionInitializer < Initializer
    %_______________________________________________________________________________________________
    
    %===============================================================================================
    properties (Constant = true, Transient = true) %                     C O N S T A N T         [P]
        multifluidCompatible = 1;
    end%CONSTANT
    
    %===============================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        %FIXME If only magnetism even worked this would be relevant...
        backgroundB;         % Initial magnetic field. Automatically actives magnetic fluxing. [1x3].
        
        waveType;            % One of the 4 MHD wave types, or 'sound' for adiabatic fluid sound.
        boxLength;
    end %PUBLIC
    
    properties(SetAccess = private, GetAccess = public)
        waveOmega;           % Omega is calculated from dispersion relation and K
        waveEigenvector;     % The wave eigenvector used to initialize the sim
    end
    
    %===============================================================================================
    properties (SetAccess = protected, GetAccess = public) %                   P R O T E C T E D [P]
        %Controlling values that define the simulation
        pBackgroundMach;      % The background fluid's velocity in units of c_s (1x1 or 3x1 dbl)
        pWavenumber;
        pAmplitude;
        pPhase; % FIXME: Not implemented yet because lazy
        % pCycles: The number of self rest frame oscillations of a travelling wave
        %          or lab rest frame cycles of a entropy wave, to run
        pCycles;
        pWaveK;               % 'Physical' K used to calculate <k|x>
        
        pDensity;
        pPressure;

        pBeLinear; % If true, uses linear wave eigenvectors; If false, uses exact characteristic
        pUseStationaryFrame; % If true, ignores pBackgroundMach and calculates the
        % background speed such that <p|x> = exactly zero

        pWriteFluid;
    end %PROTECTED
    
    properties (Dependent = true)
        amplitude; phase; backgroundMach;
        wavenumber; cycles; writeFluid;
    end
    
    %===============================================================================================
    methods %                                                                      G E T / S E T  [M]
        function obj = AdvectionInitializer(input)
            obj                 = obj@Initializer();

            obj.boxLength       = 1;
            obj.writeFluid      = 1;

            obj.pPressure       = 1;
            obj.pDensity        = 1;
            obj.backgroundMach  = [0 0 0];
            obj.backgroundB     = [0 0 0];
            

            obj.waveType        = 'entropy';
            obj.wavenumber      = 1;
            obj.amplitude       = .001;
            obj.phase           = 0;
            obj.cycles          = 2;

            obj.waveLinearity(true);
            
            obj.runCode         = 'ADVEC';
            obj.info            = 'Advection test.';
            obj.mode.fluid      = true;
            obj.mode.magnet     = false;
            obj.mode.gravity    = false;
            obj.cfl             = 0.85;
            obj.iterMax         = 1000;
            obj.ppSave.dim1     = 10;
            obj.ppSave.dim3     = 25;
            obj.activeSlices.xy = true;
            obj.activeSlices.xyz= true;
            obj.activeSlices.x  = true;
            
            obj.bcMode.x        = ENUM.BCMODE_CIRCULAR;
            obj.bcMode.y        = ENUM.BCMODE_CIRCULAR;
            obj.bcMode.z        = ENUM.BCMODE_CIRCULAR;
            
            obj.operateOnInput(input, [512 1 1]);
        end

        function set.writeFluid(self, N)
            if nargin < 2; N = 1; end
            self.pWriteFluid = N;
        end
        function N = get.writeFluid(self); N = self.pWriteFluid; end
        
        function set.amplitude(self, A)
            % Sets the nonnegative perturbation pAmplitude A
            if nargin == 1; self.pAmplitude = .001; return; end
            n = self.pWriteFluid;
            self.pAmplitude(:,n) = input2vector(A, 1, .001, false);
        end
        function A = get.amplitude(self); A = self.pAmplitude; end
        
        function set.phase(self, P)
            % Set the phase of the perturbation wave
            n = self.pWriteFluid;
            if nargin == 1; self.pPhase = 0; return; end
            self.pPhase(:,n) = input2vector(P, 1, 0, false);
        end
        function P = get.phase(self); P = self.pPhase; end

        function set.backgroundMach(self, M)
            % Set the translation speed of the background in units of Mach.
            %> V: 1-3 elements for vector mach; Default <0,0,0> for missing elements.
            n = self.pWriteFluid;
            if nargin == 1; self.pBackgroundMach(:,n) = [0; 0; 0]; return; end
            self.pBackgroundMach(:,n) = input2vector(M, 3, 0, false);
        end
        function M = get.backgroundMach(self); M = self.pBackgroundMach; end
        
        function set.wavenumber(self, V)
            % Sets the pWavenumber integer triplet; Does some input validation for us
            %> V: 1, 2, or 3 numbers; Absent elements default to <1,0,0>; noninteger is round()ed. Null vector is an error.
            n = self.pWriteFluid;
            if nargin == 1; self.pWavenumber(:,n) = [1; 0; 0]; return; end
            self.pWavenumber(:,n) = input2vector(V, 3, 0, true);
            
            if all(V == 0); error('In wavenumber(V), V evaluated to null vector! No reasonable default; wat u tryin to pull?'); end
        end
        function V = get.wavenumber(self); V = self.pWavenumber; end
        
        function set.cycles(self, C)
            % Set how many wave cycles to simulate
            % Entropy wave on stationary background takes this as sonic crossing times
            if nargin == 1; self.pCycles = 1; return; end
            self.pCycles = input2vector(C, 1, 1, false);
        end
        function C = get.cycles(self); C = self.pCycles; end
        
        function forCriticalTimes(self, NC)
            tc = 2/(self.amplitude*(self.gamma + 1));
            w_srf = norm(self.pWavenumber)*2*pi*sqrt(self.gamma*self.pPressure/self.pDensity);
            self.cycles = NC*tc/w_srf;
        end

        function setBackground(self, rho, pPressure)
            n = self.pWriteFluid;
            if nargin < 3; error('background requires (rho, pPressure) both be present.'); end
            
            self.pDensity(1,n) = input2vector(rho, 1, 1e-6, false);
            self.pPressure(1,n) = input2vector(pPressure, 1, 1e-6, false);
            
            if self.pDensity  < 0; error('Density cannot be negative!'); end
            if self.pPressure < 0; error('Pressure cannot be negative!'); end
        end

        function addNewFluid(self, copyfrom)
            self.pWriteFluid = size(self.pAmplitude, 2) + 1;

            self.amplitude      = self.pAmplitude(1,copyfrom);
            self.phase          = self.pPhase(1,copyfrom);
            self.backgroundMach = self.pBackgroundMach(:,copyfrom);
            self.wavenumber     = self.pWavenumber(:,copyfrom);
            self.setBackground(self.pDensity(1,copyfrom), self.pPressure(1,copyfrom));
            self.fluidDetails(end+1) = fluidDetailModel('10um_iron_balls');

	    self.numFluids = self.numFluids + 1; 
        end

    end%GET/SET
    
    %===============================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
        function waveLinearity(self, tf)
        % If called with true, flips linearity on: Use of infinitesmal eigenvectors and turns exact stationary frame
            if tf; self.pBeLinear = 1; self.pUseStationaryFrame = 0; else; self.pBeLinear = 0; end
        end

        function waveStationarity(self, tf)
        % If called with true, regardless of use of linear wavevector, will ignore backgroundMach and put the wave in an exactly stationary frame
            if tf; self.pUseStationaryFrame = 1; else; self.pUseStationaryFrame = 0; end
        end


    end%PUBLIC
    
    %===============================================================================================
    methods (Access = protected) %                                          P R O T E C T E D    [M]
        function [fluids, mag, statics, potentialField, selfGravity] = calculateInitialConditions(obj)
            potentialField = [];
            selfGravity = [];
            
            geo = obj.geomgr;
            
            statics  = StaticsInitializer(geo);
            
            geo.makeBoxSize(obj.boxLength);
            geo.makeBoxOriginCoord(obj.boxLength * [-.5 -.5 -.5]);
            
            [xGrid, yGrid, zGrid] = geo.ndgridSetIJK('pos');

            tMax = zeros([obj.numFluids 1]);
            
            if obj.numFluids > 1
                % zeros are gas velocity then dust velocity
                dragConstant = dustyBoxDragTime(obj.fluidDetails, obj.pDensity(1), obj.pDensity(2), 0, 0, obj.gamma(1), obj.pPressure(1));
            end
            
            for fluidCt = 1:obj.numFluids            
                % Calculate the background velocity
                % FIXME HACK HACK HACK using wrong EoS if fluidCt != 1...
                % FIXME note that P = rho kb T / mu extremely small if mu is dust-like
                %c_s      = sqrt(obj.gamma(1,1)*obj.pPressure(1,fluidCt)/obj.pDensity(1,fluidCt)  ); % The infinitesmal soundspeed of gas'
                c_s      = sqrt(obj.gamma(1,1)*obj.pPressure(1,1)/obj.pDensity(1,1)  ); % The infinitesmal soundspeed of gas

                if obj.pAmplitude(1,1) ~= 0
                    % omega = c_wave k
                    % \vec{k} = \vec{N} * 2pi ./ \vec{L} = \vec{N} * 2pi ./ [1 ny/nx nz/nx]
                    rez   = geo.globalDomainRez;
                    K     = 2*pi*obj.pWavenumber(:,1) ./ ([1; rez(2)/rez(1); rez(3)/rez(1)] * obj.boxLength);
                    obj.pWaveK(:,fluidCt) = K;
                    KdotX = K(1)*xGrid + K(2)*yGrid + K(3)*zGrid; % K.X is used much.
                    
                    % For debugging, this plops down a finite subset of the oscillating waveform
                    % so we have spatial locality to distinguish between forward/static/backward going
                    % waves
                    %KdotX = K(2)*yGrid + K(3)*zGrid; % K.X is used much
                    %pik = ((xGrid > .25) & (xGrid < .5));
                    %KdotX(pik) = K(1)*xGrid(pik)+ K(2)*yGrid(pik) + K(3)*zGrid(pik); % K.X is used much.
                    
                    if obj.pUseStationaryFrame
                        bgvelocity = -c_s*finampRelativeSoundspeed(obj.pAmplitude(1,fluidCt), obj.gamma(1,fluidCt))*K/norm(K);
                    else
                        bgvelocity = c_s * obj.pBackgroundMach;
                    end
                    
                    FW = FluidWaveGenerator(obj.pDensity, bgvelocity, obj.pPressure, obj.backgroundB, obj.gamma);
                    
                    % Someday we'll do magnetic fields again...
                    if strcmp(obj.waveType, 'sonic') && norm(obj.backgroundB) > 0
                        obj.waveType = 'fast ma';
                        disp('WARNING: sonic wave selected with nonzero B. Changing to fast MA.');
                    end
                    
                    if strcmp(obj.waveType, 'entropy')
                        amp = abs(obj.pAmplitude(1,fluidCt)) * cos(KdotX + obj.pPhase(1,1));
                        [omega, evector] = FW.entropyExact(amp, K);
                        omega = norm(obj.pWaveK(1,fluidCt))*sqrt(obj.gamma(1,fluidCt));
                    elseif strcmp(obj.waveType, 'sonic')
                        
                        if obj.numFluids == 1
                            amp = abs(obj.pAmplitude(1,1)) * cos(KdotX + obj.pPhase(1,1));
                            if obj.pBeLinear
                                [omega, evector] = FW.sonicInfinitesmal(amp, K);
                            else
                                [omega, evector] = FW.sonicExact(amp, K);
                            end
                        else
                            [omega, evector] = FW.dustyLinear(abs(obj.pAmplitude(1,1)), KdotX + obj.pPhase(1,1), K, fluidCt, dragConstant, 1);
                        end
                    elseif strcmp(obj.waveType, 'dustydamp')
                        if obj.numFluids ~= 2
                            error(['Fatal: ''dustydamp'' selected as wave type, but have ' num2str(self.numFluids) ' fluids, not exactly two fluids.']);
                        end
                        % This is the third nontrivial eigenvalue of DustyWave,
                        % This mode is evanescent and not generally of interest...
                        amp = abs(obj.pAmplitude(1,fluidCt)) * cos(KdotX + obj.pPhase(1,fluidCt));
                        [omega, evector] = FW.dustyLinear(amp, K, fluidCt, 0);
                    elseif strcmp(obj.waveType, 'fast ma')
                        [evector, omega] = eigenvectorMA(1, c_s^2, velocity, B0, K, 2);
                        evector(8) = c_s^2 * evector(1);
                    end
                    
                    mass = FW.waveRho;
                    vel  = FW.waveVelocity();
                    ener = FW.waveInternalEnergy();
                else
                    [mass, vel, ~, ener] = geo.basicFluidXYZ();
                    mass = mass * obj.pDensity(1,fluidCt);

                    for q = 1:3; vel(q,:,:,:) = obj.pBackgroundMach(q,fluidCt) * c_s; end
                    ener = ener * obj.pPressure(1,fluidCt) / (obj.fluidDetails(fluidCt).gamma-1);
                    
                    omega = 2*pi; % HACK HACK HACK
                    evector = [1 0 0 1 0];
                    K = [1 0 0]; % HACK HACK HACK
                end

                fluids(fluidCt)         = obj.rhoVelEintToFluid(mass, vel, ener);
                fluids(fluidCt).details = obj.fluidDetails(fluidCt);

                obj.waveOmega = omega;
                obj.waveEigenvector = evector;
                wavespeed     = omega / norm(K);
            
                % forward speed = background speed + wave speed; Sim time = length/speed, length \eq 1
                if abs(wavespeed) < .05*c_s; wavespeed = c_s; end
                tMax(fluidCt)  = obj.pCycles / (abs(wavespeed)*norm(obj.pWavenumber));
            end

            mag = zeros([3 geo.localDomainRez]);
            
            if max(abs(obj.backgroundB)) > 0
                obj.mode.magnet = true; obj.cfl = .4; obj.pureHydro = 0;
            else
                obj.pureHydro = 1;
            end
            
            obj.timeMax = max(tMax);
            SaveManager.logPrint('Running wave type: %s\nWave speed in simulation frame: %f\n', obj.waveType, wavespeed);
        end
    end%PROTECTED
    
    %===============================================================================================
    methods (Static = true) %                                                     S T A T I C    [M]
    end%PROTECTED
    
end%CLASS
    
