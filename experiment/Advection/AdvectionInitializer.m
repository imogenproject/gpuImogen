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

        waveK;               % Wavenumber of wave existing on periodic domain
                             % Omega is calculated from dispersion relation.
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
            obj.waveK           = [1 0 0];
            obj.waveAmplitude   = .01;
            obj.waveDirection   = 1;

            obj.runCode         = 'ADVEC';
            obj.info            = 'Advection test.';
            obj.mode.fluid      = true;
            obj.mode.magnet     = false;
            obj.mode.gravity    = false;
            obj.cfl             = 0.7;
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

            mom(1,:,:,:) = mass*velocity(1);
            mom(2,:,:,:) = mass*velocity(2);
            mom(3,:,:,:) = mass*velocity(3);
            ener         = ener + .5*(squeeze(sum(mom.^2,1))./mass + .5*squeeze(sum(mag.^2,1)));

            % omega = c_wave k
            % \vec{k} = \vec{N} * 2pi ./ \vec{L} = \vec{N} * 2pi ./ [1 ny/nx nz/nx]
            K     = 2*pi*obj.waveK ./ obj.grid
            B0    = obj.backgroundB;
            phase = K(1)*xGrid + K(2)*yGrid + K(3)*zGrid; % K.X is used much.
            
            if obj.waveDirection >= 0; obj.waveDirection = 1; else; obj.waveDirection = -1; end

            % Depending on the wave selected, we calculate the appropriate wave eigenvector
            % Allow dEpsilon to represent the perturbation in pressure
            % The quantities are then appended to the equilibrium flow strictly to 1st order as is
            % appropriate for a linear wave calculation.
            if strcmp(obj.waveType, 'entropy');
                drho = obj.waveAmplitude * sin(phase);
                dv = zeros(size(mom));
                dEpsilon = 0; dB = 0;

                omega = (K*velocity');
                wavespeed = omega / norm(K);
            elseif strcmp(obj.waveType, 'sound')
                if max(abs(obj.backgroundB)) > 0; error('Sonic wave simulation requires zero magnetic field.'); end
                drho = obj.waveAmplitude*sin(phase);
                dv = zeros(size(mom));
                
                for i = 1:3;
                    dv(i,:,:,:) = obj.waveDirection*c_s*K(i)*drho./(mass*norm(K));
                end
                dEpsilon = c_s^2 * drho; dB = 0;

                omega = K*velocity' + c_s*norm(K);
                wavespeed = omega / norm(K);
            elseif strcmp(obj.waveType, 'alfven')
                if max(abs(obj.backgroundB)) == 0; error('Alfven wave simulation requires nonzero magnetic field'); end

                error('Not programmed in yet');
            elseif strcmp(obj.waveType, 'fast ma') || strcmp(obj.waveType, 'slow ma')
                if max(abs(obj.backgroundB)) == 0; error('MA wave simulation requires nonzero magnetic field'); end

                ksq = K * K';
                bsq = B0 * B0';
                kdb = K * B0';
                
                lambda = sort(solveQuartic(1, 0, -ksq*(bsq + c_s^2), 0, ksq*c_s^2*kdb^2));
                if strcmp(obj.waveType, 'fast ma')
                    if obj.waveDirection == 1; lambda = lambda(1); else; lambda = lambda(4); end
                else
                    if obj.waveDirection == 1; lambda = lambda(2); else; lambda = lambda(3); end
                end
                omega = K*velocity' - lambda;
                
                flow.rho = 1; flow.gamma = 5/3; flow.P = obj.pressure;
                flow.vx = velocity(1); flow.vy = velocity(2); flow.vz = velocity(3);
                flow.bx = B0(1);       flow.by = B0(2);       flow.bz = B0(3);
                
                ev = eigenvectorMA(flow, K(1), K(2), K(3), omega);
                
                drho = obj.waveAmplitude*sin(phase);
                for i = 1:3;
                    dv(i,:,:,:) = obj.waveAmplitude*abs(ev(1+i))*sin(phase + angle(ev(1+i)));
                    dB(i,:,:,:) = obj.waveAmplitude*abs(ev(4+i))*sin(phase + angle(ev(4+i)));
                end
                dEpsilon = c_s*drho;
                
                wavespeed = omega / norm(K);
            else
                error(sprintf('Wave type %s is not a recognized form; Aborting.', obj.waveType));
            end

            % We now begin calculating linear wave equations.
            % Note that the following transforms are necessary to convert from the {drho, dv, dP} basis
            % to the {drho, dp, dE} basis
            %     rho <- rho + drho
            %     p   <- p + v drho + rho dv
            %     E   <- E + (d[.5 rho v^2]     ) + d[P] / (gamma-1)
            %   = E   <- E + (.5 v^2 drho + p dv) + c_s^2 drho / (gamma-1)
            ener = ener + .5*speed^2*drho + squeeze(sum(mom.*dv + mag.*dB,1)) + dEpsilon / (obj.gamma - 1);
            for i = 1:3
                mom(i,:,:,:) = squeeze(mom(i,:,:,:)).*(1 + drho./mass) + mass.*squeeze(dv(i,:,:,:));
            end
            mag  = mag  + dB;
            mass = mass + drho;

            % forward speed = background speed + wave speed. Time = length / speed
            if abs(wavespeed) < .05*c_s; wavespeed = c_s; end
            obj.timeMax  = obj.numWavePeriods / abs(wavespeed);

            if max(abs(obj.backgroundB)) > 0; obj.mode.magnet = true; obj.cfl = .4; obj.pureHydro = 0; else; obj.pureHydro = 1; end

            fprintf('Running wave type: %s\nWave speed in simulation frame: %f\n', obj.waveType, wavespeed);
        end
        
        end%PROTECTED
                
%===================================================================================================        
        methods (Static = true) %                                                                   S T A T I C    [M]
        end%PROTECTED
        
end%CLASS
    
