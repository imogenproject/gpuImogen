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
        backgroundB;         % Initial magnetic field. Automatically actives magnetic fluxing. [3x1].
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
            obj.cfl             = 0.4;
            obj.iterMax         = 1000;
            obj.ppSave.dim1     = 10;
            obj.ppSave.dim3     = 25;
            obj.activeSlices.xy = true;
            obj.activeSlices.xyz= true;
            obj.activeSlices.x  = true;
            
            obj.bcMode          = ENUM.BCMODE_CIRCULAR;
            
            obj.operateOnInput(input);
        end
        
        end%GET/SET
        
%===================================================================================================
    methods (Access = public) %                                                                      P U B L I C  [M]
        end%PUBLIC
        
%===================================================================================================        
        methods (Access = protected) %                                                              P R O T E C T E D    [M]
        
        function [mass, mom, ener, mag, statics] = calculateInitialConditions(obj)
            statics  = StaticsInitializer(obj.grid);

            obj.dGrid = ones(1,3) / obj.grid(1); % set the total grid length to be 1. 

            [xGrid yGrid zGrid] = ndgrid(2*pi*(1:obj.grid(1))/obj.grid(1), ...
                                         2*pi*(1:obj.grid(2))/obj.grid(2), ...
                                         2*pi*(1:obj.grid(3))/obj.grid(3));

            % Store equilibrium parameters
            mass     = 1*ones(obj.grid);
            mom      = zeros([3, obj.grid]);
            mag      = zeros([3, obj.grid]);
            ener     = ones(obj.grid)*obj.pressure/(obj.gamma-1); 

            c_s      = sqrt(obj.gamma*obj.pressure);
            for i = 1:3; mag(i,:,:,:) = obj.backgroundB(i); end
            
            % Add velocity boost            
            velocity    = c_s * obj.backgroundMach;
            if numel(velocity) == 1; velocity = [velocity 0 0]; end
            speed = norm(velocity);

            mom(1,:,:,:) = mass*velocity(1);
            mom(2,:,:,:) = mass*velocity(2);
            mom(3,:,:,:) = mass*velocity(3);
            ener         = ener + .5*squeeze(sum(mom.^2,1))./mass;

            % omega = c_wave k
            % \vec{k} = \vec{N} * 2pi ./ \vec{L} = \vec{N} * 2pi ./ [1 ny/nx nz/nx]
            K     = obj.grid(1) * obj.waveK * 2 * pi ./ obj.grid;
            phase = obj.waveK(1)*xGrid + obj.waveK(2)*yGrid + obj.waveK(3)*zGrid; % K.X is used much.
            
            if obj.waveDirection >= 0; obj.waveDirection = 1; else; obj.waveDirection = -1; end

            % Depending on the wave selected, we calculate the appropriate wave eigenvector
            % Allow dEpsilon to represent the perturbation in pressure
            % The quantities are then appended to the equilibrium flow strictly to 1st order as is
            % appropriate for a linear wave calculation.
            if strcmp(obj.waveType, 'entropy');
                drho = obj.waveAmplitude * sin(phase);
                dv = zeros(size(mom));
                dEpsilon = 0;

                omega = (K*velocity');
                wavespeed = omega / K(1);
            elseif strcmp(obj.waveType, 'sound')
                drho = obj.waveAmplitude*sin(phase);
                dv = zeros(size(mom));
                
                for i = 1:3;
                    dv(i,:,:,:) = obj.waveDirection*c_s*K(i)*drho./(mass*norm(K));
                end
                dEpsilon = c_s^2 * drho;

                omega = K*velocity' + c_s*norm(K);
                wavespeed = omega / K(1);
            elseif strcmp(obj.waveType, 'alfven')


                error('Not programmed in yet');
            elseif strcmp(obj.waveType, 'fast ma')
                error('Not programmed in yet');
            elseif strcmp(obj.waveType, 'slow ma')
                error('Not programmed in yet');
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
            ener = ener + .5*speed^2*drho + squeeze(sum(mom.*dv,1)) + dEpsilon / (obj.gamma - 1);
            for i = 1:3
                mom(i,:,:,:) = squeeze(mom(i,:,:,:)).*(1 + drho./mass) + mass.*squeeze(dv(i,:,:,:));
            end
            mass = mass + drho;

            % forward speed = background speed + wave speed. Time = length / speed
            if abs(wavespeed) < .05*c_s; wavespeed = c_s; end
            obj.timeMax  = obj.numWavePeriods / abs(wavespeed); % Selected # of complete advections in the X direction.

            if max(abs(obj.backgroundB)) > 0; obj.modes.magnet = true; end

            fprintf('Running wave type: %s\nWave x speed in simulation frame: %f\n', obj.waveType, wavespeed);
        end
        
        end%PROTECTED
                
%===================================================================================================        
        methods (Static = true) %                                                                   S T A T I C    [M]
        end%PROTECTED
        
end%CLASS
    
