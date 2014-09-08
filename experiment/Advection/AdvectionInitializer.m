classdef AdvectionInitializer < Initializer
%___________________________________________________________________________________________________ 
        
%===================================================================================================
        properties (Constant = true, Transient = true) %                                            C O N S T A N T         [P]
    end%CONSTANT
        
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                                           P U B L I C  [P]
        backgroundMach;      % The background fluid's velocity in dimensionless terms.
                             % 3x1 double: velocity
                             % scalar    : equivalent [x speed, 0, 0] 
        backgroundB;         % Initial magnetic field. Automatically actives magnetic fluxing. [1x3].
        density;
        pressure;

        numWavePeriods;      % The number of periods to run for (i.e. number of times to advect it through the box)
        waveType;            % One of the 4 MHD wave types, or 'sound' for adiabatic fluid sound.

        waveN;               % Wavenumber of wave existing on periodic domain
        waveK;               % 'Physical' K used to calculate <k|x>
        waveOmega;           % Omega is calculated from dispersion relation.
                             
        waveAmplitude;
        waveDirection;       % >= 0 or < 0 for the forward or backward propagating wave respectively

    end %PUBLIC

%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                                     P R O T E C T E D [P]
    end %PROTECTED
        
        
%===================================================================================================
    methods %                                                                                       G E T / S E T  [M]
        
        function obj = AdvectionInitializer(input)
            obj                 = obj@Initializer();
            obj.gamma           = 5/3;
            obj.pressure        = 1;
            obj.density         = 1;
            obj.backgroundMach  = [.5 0 0];
            obj.backgroundB     = [0 0 0];

            obj.numWavePeriods     = 2;

            obj.waveType        = 'entropy';
            obj.waveN           = [1 0 0];
            obj.waveAmplitude   = .01;
            obj.waveDirection   = 1;

            obj.runCode         = 'ADVEC';
            obj.info            = 'Advection test.';
            obj.mode.fluid      = true;
            obj.mode.magnet     = false;
            obj.mode.gravity    = false;
            obj.cfl             = 0.35;
            obj.iterMax         = 1000;
            obj.ppSave.dim1     = 10;
            obj.ppSave.dim3     = 25;
            obj.activeSlices.xy = true;
            obj.activeSlices.xyz= true;
            obj.activeSlices.x  = true;
            
            obj.bcMode.x          = ENUM.BCMODE_CIRCULAR;
            obj.bcMode.y          = ENUM.BCMODE_CIRCULAR;
            obj.bcMode.z          = ENUM.BCMODE_CIRCULAR;

            
            obj.operateOnInput(input);
        end
        
        end%GET/SET
        
%===================================================================================================
    methods (Access = public) %                                                                      P U B L I C  [M]
        end%PUBLIC
        
%===================================================================================================        
        methods (Access = protected) %                                                              P R O T E C T E D    [M]
        
        function [mass, mom, ener, mag, statics, potentialField, selfGravity] = calculateInitialConditions(obj)
            statics  = StaticsInitializer();
            potentialField = [];
            selfGravity = [];

            GIS = GlobalIndexSemantics();

            obj.dGrid = 1 ./ obj.grid; % set the total grid length to be 1. 

            [xGrid yGrid zGrid] = GIS.ndgridSetXYZ();
            xGrid = xGrid/obj.grid(1);
            yGrid = yGrid/obj.grid(2);
            zGrid = zGrid/obj.grid(3);

            % Store equilibrium parameters
            mass     = 1*ones(GIS.pMySize);
            mom      = zeros([3, GIS.pMySize]);
            mag      = zeros([3, GIS.pMySize]);
            ener     = ones(GIS.pMySize)*obj.pressure/(obj.gamma-1); 

            c_s      = sqrt(obj.gamma*obj.pressure);
            for i = 1:3; mag(i,:,:,:) = obj.backgroundB(i); end
            
            % Add velocity boost            
            velocity    = c_s * obj.backgroundMach;
            if numel(velocity) == 1; velocity = [velocity 0 0]; end
            speed = norm(velocity);

            mom(1,:,:,:) = velocity(1);
            mom(2,:,:,:) = velocity(2);
            mom(3,:,:,:) = velocity(3);

            % omega = c_wave k
            % \vec{k} = \vec{N} * 2pi ./ \vec{L} = \vec{N} * 2pi ./ [1 ny/nx nz/nx]
            K     = 2*pi*obj.waveN ./ [1 obj.grid(2)/obj.grid(1) obj.grid(3)/obj.grid(1)]; obj.waveK = K;
            B0    = obj.backgroundB;
            KdotX = K(1)*xGrid + K(2)*yGrid + K(3)*zGrid; % K.X is used much.
            hgrid = obj.dGrid.*K;
            phase = 0;
            
            if obj.waveDirection >= 0; obj.waveDirection = 1; else; obj.waveDirection = -1; end

            % Depending on the wave selected, we calculate the appropriate wave eigenvector
            % Allow dEpsilon to represent the perturbation in pressure
            % The quantities are then appended to the equilibrium flow strictly to 1st order as is
            % appropriate for a linear wave calculation.
            if strcmp(obj.waveType, 'sonic') && norm(B0) > 0
                obj.waveType = 'fast ma';
                disp('WARNING: sonic wave selected with nonzero B. Changing to fast MA.');
            end

            if strcmp(obj.waveType, 'entropy')
                [waveEigenvector omega] = eigenvectorEntropy(1, c_s^2, velocity, B0, K);
                waveEigenvector(8) = 0;
            elseif strcmp(obj.waveType, 'sonic')
                [waveEigenvector omega] = eigenvectorSonic(1, c_s^2, velocity, B0, K, obj.waveDirection);
                waveEigenvector(8) = c_s^2 * waveEigenvector(1);
            elseif strcmp(obj.waveType, 'fast ma')
                [waveEigenvector omega] = eigenvectorMA(1, c_s^2, velocity, B0, K, 2*obj.waveDirection);
                waveEigenvector(8) = c_s^2 * waveEigenvector(1);
            end

            [drho dV dB deps] = evaluateWaveEigenvector(exp(1i*phase) * obj.waveAmplitude * waveEigenvector, ...
                                KdotX, hgrid);
            wavespeed = omega / norm(K);
            obj.waveOmega = omega;

            % Note that the following transforms are necessary to convert from the {drho, dv, dP} basis
            % to the {drho, dp, dE} basis
            %     rho <- rho + drho
            %     p   <- p + v drho + rho dv
            %     E   <- E + (d[.5 rho v^2]     ) + d[P] / (gamma-1)
            %   = E   <- E + (.5 v^2 drho + p dv) + c_s^2 drho / (gamma-1)
            mass = mass + drho;
	    for i = 1:3
                mom(i,:,:,:) = squeeze(mom(i,:,:,:) + dV(i,:,:,:)).*mass; % p = v*(rho+drho)+dv(rho+drho)==p + v*drho + rho*dv +drho*dv
	    end
            mag  = mag  + dB;
            ener = ener + deps / (obj.gamma - 1) + 0.5*squeeze(sum(mom.^2,1))./mass + 0.5*squeeze(sum(mag.^2,1));

            % forward speed = background speed + wave speed; Sim time = length/speed
            if abs(wavespeed) < .05*c_s; wavespeed = c_s; end
            obj.timeMax  = obj.numWavePeriods / (abs(wavespeed)*norm(obj.waveN));

            if max(abs(obj.backgroundB)) > 0;
                obj.mode.magnet = true; obj.cfl = .4; obj.pureHydro = 0;
            else;
                obj.pureHydro = 1;
            end

            fprintf('Running wave type: %s\nWave speed in simulation frame: %f\n', obj.waveType, wavespeed);
        end
        
        end%PROTECTED
                
%===================================================================================================        
        methods (Static = true) %                                                                   S T A T I C    [M]
        end%PROTECTED
        
end%CLASS
    
