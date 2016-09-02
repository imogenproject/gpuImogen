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
    end %PUBLIC
    
    properties(SetAccess = private, GetAccess = public);
        waveOmega;           % Omega is calculated from dispersion relation and K
    end
    
    %===============================================================================================
    properties (SetAccess = protected, GetAccess = public) %                   P R O T E C T E D [P]
        %Controlling values that define the simulation
        pBackgroundMach;      % The background fluid's velocity in multiples of infinitesmal c_s (3x1 dbl)
        pWavenumber;
        pAmplitude;
        pPhase; % FIXME: Not implemented yet because lazy
        pCycles;          % The number of periods to run for (i.e. number of times to advect it through the box)
        pWaveK;               % 'Physical' K used to calculate <k|x>
        
        pDensity;
        pPressure;

        pBeLinear; % If true, uses infinitesmal wave eigenvectors; If false, uses exact characteristics
        pUseStationaryFrame; % If true, ignores pBackgroundMach and calculates the exact translational velocity
        % to keep the wave exactly stationary

        pTwoFluidMode;
    end %PROTECTED
    
    properties (Dependent = true)
        amplitude; backgroundMach; wavenumber; cycles;
    end
    
    %===============================================================================================
    methods %                                                                      G E T / S E T  [M]
        function obj = AdvectionInitializer(input)
            obj                 = obj@Initializer();
            obj.gamma           = 5/3;
            obj.pPressure        = 1;
            obj.pDensity         = 1;
            obj.backgroundMach  = [0 0 0];
            obj.backgroundB     = [0 0 0];
            
            obj.waveType        = 'entropy';
            obj.wavenumber      = 1;
            obj.amplitude       = .001;
            obj.pPhase = 0; % FIXME: not put in yet because redundant for tests & lazy
            obj.cycles          = 2;

            obj.waveLinearity(true);

            
            obj.runCode         = 'ADVEC';
            obj.info            = 'Advection test.';
            obj.mode.fluid      = true;
            obj.mode.magnet     = false;
            obj.mode.gravity    = false;
            obj.cfl             = 0.45;
            obj.iterMax         = 1000;
            obj.ppSave.dim1     = 10;
            obj.ppSave.dim3     = 25;
            obj.activeSlices.xy = true;
            obj.activeSlices.xyz= true;
            obj.activeSlices.x  = true;
            
            obj.bcMode.x          = ENUM.BCMODE_CIRCULAR;
            obj.bcMode.y          = ENUM.BCMODE_CIRCULAR;
            obj.bcMode.z          = ENUM.BCMODE_CIRCULAR;
            
            obj.operateOnInput(input, [512 1 1]);
        end
        
        function set.amplitude(self, A)
            % Sets the nonnegative perturbation pAmplitude A
            if nargin == 1; self.pAmplitude = .001; return; end;
            self.pAmplitude = input2vector(A, 1, .001, false);
        end
        function A = get.amplitude(self); A = self.pAmplitude; end
        
        function set.backgroundMach(self, M)
            % Set the translation speed of the background in units of Mach.
            %> V: 1-3 elements for vector mach; Default <0,0,0> for missing elements.
            if nargin == 1; self.pBackgroundMach = [0 0 0]; return; end
            self.pBackgroundMach = input2vector(M, 3, 0, false);
        end
        function M = get.backgroundMach(self); M = self.pBackgroundMach; end
        
        function set.wavenumber(self, V)
            % Sets the pWavenumber integer triplet; Does some input validation for us
            %> V: 1, 2, or 3 numbers; Absent elements default to <1,0,0>; noninteger is round()ed. Null vector is an error.
            if nargin == 1; self.pWavenumber = [1 0 0]; return; end
            self.pWavenumber = input2vector(V, 3, 0, true);
            
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
            if nargin < 3; error('background requires (rho, pPressure) both be present.'); end
            
            self.pDensity = input2vector(rho, 1, 1e-6, false);
            self.pPressure = input2vector(pPressure, 1, 1e-6, false);
            
            if self.pDensity  < 0; error('Density cannot be negative!'); end;
            if self.pPressure < 0; error('Pressure cannot be negative!'); end;
        end

        function setTwoFluidMode(self, tf)
            if tf;
                self.pTwoFluidMode = 1;
                fprintf('WARNING: ACTIVATING MULTIPHASE FLUID SIMULATION\n');
                fprintf('WARNING: THIS IS HIGHLY EXPERIMENTAL.\n');
            else
                self.pTwoFluidMode = 0;
            end
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
            
            geo.makeBoxSize(1);
            geo.makeBoxOriginCoord([-.5 -.5 -.5]);
            
            [xGrid yGrid zGrid] = geo.ndgridSetIJK('pos');
            
            fluids = struct('mass',[],'momX',[],'momY',[],'momZ',[],'ener',[]);

            % Store equilibrium parameters
            [mass mom mag ener] = geo.basicFluidXYZ();
            mass     = mass * obj.pDensity;
            ener     = ener*obj.pPressure / (obj.gamma-1);
            
            % Assert the background magnetic field
            for i = 1:3; mag(i,:,:,:) = obj.backgroundB(i); end

            % omega = c_wave k
            % \vec{k} = \vec{N} * 2pi ./ \vec{L} = \vec{N} * 2pi ./ [1 ny/nx nz/nx]
            rez   = geo.globalDomainRez;
            K     = 2*pi*obj.pWavenumber ./ [1 rez(2)/rez(1) rez(3)/rez(1)]; obj.pWaveK = K;
            KdotX = K(1)*xGrid + K(2)*yGrid + K(3)*zGrid; % K.X is used much.
            
            % Calculate the background velocity
            c_s      = sqrt(obj.gamma*obj.pPressure); % The infinitesmal soundspeed
            if obj.pUseStationaryFrame;
                bgvelocity = -c_s*finampRelativeSoundspeed(obj.pAmplitude, obj.gamma)*K/norm(K);
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
                amp = abs(obj.pAmplitude) * cos(KdotX + angle(obj.pAmplitude));
                FW.entropyExact(amp, K);
            elseif strcmp(obj.waveType, 'sonic')
                amp = abs(obj.pAmplitude) * cos(KdotX + angle(obj.pAmplitude));
                if obj.pBeLinear; FW.sonicInfinitesmal(amp, K); else
                                  FW.sonicExact(amp, K); end

                omega = 2*pi*sqrt(obj.gamma); % FIXME do not assume this is normalized
            elseif strcmp(obj.waveType, 'fast ma')
                [waveEigenvector omega] = eigenvectorMA(1, c_s^2, velocity, B0, K, 2);
                waveEigenvector(8) = c_s^2 * waveEigenvector(1);
            end

            mass = FW.waveRho;
            mom  = FW.waveMomentum();
            ener = FW.waveTotalEnergy();
            
            obj.waveOmega = omega;
            wavespeed = omega / norm(K);
            
            % forward speed = background speed + wave speed; Sim time = length/speed, length \eq 1
            if abs(wavespeed) < .05*c_s; wavespeed = c_s; end
            obj.timeMax  = obj.pCycles / (abs(wavespeed)*norm(obj.pWavenumber));
            
            if max(abs(obj.backgroundB)) > 0;
                obj.mode.magnet = true; obj.cfl = .4; obj.pureHydro = 0;
            else;
                obj.pureHydro = 1;
            end
            
            if obj.pTwoFluidMode
                fluids(1).mass = mass; fluids(1).momX = squish(mom(1,:,:,:));
                fluids(1).momY = squish(mom(2,:,:,:)); fluids(1).momZ = squish(mom(3,:,:,:));
                fluids(1).ener = ener;
		fluids(1).details = [];

                fluids(2).mass = mass; fluids(2).momX = -squish(mom(1,:,:,:));
                fluids(2).momY = -squish(mom(2,:,:,:)); fluids(2).momZ = -squish(mom(3,:,:,:));
                fluids(2).ener = ener;
		fluids(2).details = [];
                fprintf('WARNING: Experimental two-fluid mode: Generating 2nd fluid with reversed momentum!\n');
            else
                fluids(1).mass = mass; fluids(1).momX = squish(mom(1,:,:,:));
                fluids(1).momY = squish(mom(2,:,:,:)); fluids(1).momZ = squish(mom(3,:,:,:));
                fluids(1).ener = ener;
		fluids(1).details = [];
            end

            if mpi_amirank0(); fprintf('Running wave type: %s\nWave speed in simulation frame: %f\n', obj.waveType, wavespeed); end
        end
    end%PROTECTED
    
    %===============================================================================================
    methods (Static = true) %                                                     S T A T I C    [M]
    end%PROTECTED
    
end%CLASS
    
