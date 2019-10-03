classdef NuclearExplosion < Initializer
% Run a simulation of the RT instability to test Imogen
%
%   useStatics        specifies if static conditions should be set for the run.       logical  
        
%===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]
        RHO_CONST = 1; % sets constant density atmosphere, T ~ 1 - integral(rho G)
        RHO_GRAD  = 2; % sets linear density gradient
        RHO_EXP   = 3; % sets exponential density falloff, T ~ constant
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
    gravConstant;
    rhoMode;
    rho0, rho1;
    P0;
    Bx;
    Bz;
    Kx;
    Ky;
    Kz;
    pertAmplitude;
    randomPert;

    end %PUBLIC

%===================================================================================================
    properties (Dependent = true) %                                            D E P E N D E N T [P]

    end %DEPENDENT
    
%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]

    end %PROTECTED

%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        
%____________________________________________________________________________ NuclearExplosion
        function obj = NuclearExplosion(input)            
            obj                     = obj@Initializer();
            obj.runCode             = 'RAYLEIGH_TAYLOR';
            obj.info                = 'Rayleigh-Taylor instability test';
            obj.mode.fluid          = true;
            obj.pureHydro          = true;        
            obj.mode.magnet         = false;
            obj.mode.gravity        = false;
            obj.iterMax             = 300;
            obj.fluidDetails(1).gamma = 1.4;
            obj.bcMode.x            = 'circ';
            obj.bcMode.y            = 'mirror';
            obj.bcMode.z            = 'circ';

            obj.activeSlices.xy     = true;
            obj.gravConstant        = 1;
            obj.gravity.constant    = .1;
            obj.gravity.solver      = ENUM.GRAV_SOLVER_EMPTY;

            obj.rhoMode             = obj.RHO_CONST;
            obj.rho0                = 1;
            obj.rho1                = -.25;
            obj.P0                  = 2.5;
            obj.Bx                  = 0;
            obj.Bz                  = 0;

            obj.pertAmplitude = .0001;
            obj.Kx = 1;
            obj.Ky = 1;
            obj.Kz = 1;
            obj.randomPert = 1;
            
            obj.operateOnInput(input, [200, 100, 100]);
        end
        
    end%GET/SET
    
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
    end%PUBLIC
    
%===================================================================================================    
    methods (Access = protected) %                                          P R O T E C T E D    [M]                
        
%________________________________________________________________________ calculateInitialConditions
        function [fluids, mag, statics, potentialField, selfGravity] = calculateInitialConditions(obj)

            potentialField = PotentialFieldInitializer();
            selfGravity = [];
            statics = [];

            geo = obj.geomgr;
            grid = geo.globalDomainRez;
            
            if grid(2) == 1; obj.bcMode.y = ENUM.BCMODE_CIRCULAR; end
            if grid(3) == 1; obj.bcMode.z = ENUM.BCMODE_CIRCULAR; end
            
            geo.makeBoxSize(grid(2)/grid(1));

            X = geo.localXposition;
            Y = geo.localYposition;
            Z = geo.localZposition;
            [xp, yp, zp] = ndgrid(X, Y, Z);

            % Initialize Arrays
            [mass, mom, mag, ener] = geo.basicFluidXYZ();

            % Setup density according to the rhoMode variable,
            % and set gas pressure according to grad P = rho g
            switch obj.rhoMode
                case obj.RHO_CONST
                    mass = mass*obj.rho0;
                    ener = obj.P0 - obj.gravConstant * yp * obj.rho0;
                case obj.RHO_GRAD
                    mass = obj.rho0 - obj.rho1 * yp;
                    ener = obj.P0 - obj.gravConstant * yp .* (obj.rho0 - obj.rho1 * yp/2);
                case obj.RHO_EXP
                    mass = obj.rho0*exp(-obj.rho1 * yp * obj.gravConstant);
                    ener = (obj.rho0 * obj.gravConstant / obj.rho1) * exp(-obj.rho1 * yp);
            end
            
            ener(round(end/2), 32, round(end/2)) = 10000;
            
            % Create gravity field
            
            potentialField.field = yp;
            potentialField.constant = obj.gravConstant;
            if yp(1,1,1) == 0
                potentialField.field(:,1:4,:) = -potentialField.field(:,1:4,:);
            end

        % If random perturbations are selected, it will impart random y-velocity to each column of the grid.
        % Otherwise, it will create a sinusoid of wavenumber Kx,Ky,Kz.
%            if obj.randomPert == 0
%                if geo.localDomainRez(3) == 1
%                    mom(2,:,:,:) = cos(2*pi*obj.Kx*xp) .* exp(-2*abs(yp-Y0)*obj.Kx*2*pi);
%                else
%                    mom(2,:,:,:) = cos(2*pi*obj.Kx*xp) .* cos(2*pi*obj.Kz*zp) .* exp(-2*abs(yp-Y0)*obj.Kx*2*pi);
%                end
%            else
%                w = (rand([geo.localDomainRez(1) geo.localDomainRez(3)])*-0.5);
%
%                for q = 1:geo.localDomainRez(2); mom(2,:,q,:) = w * exp(-2*abs(Y(q)-Y0)*obj.Kx*2*pi); end
%            end
%            mom(2,:,:,:) = squish(mom(2,:,:,:)).*mass*obj.pertAmplitude;

            % Don't perturb +y limit
	    if geo.edgeInterior(2,2) == 0; mom(2,:,(end-2):end,:) = 0; end
        
            % If doing magnetic R-T, turn on magnetic flux & set magnetic field & add magnetic energy
            if (obj.Bx ~= 0.0) || (obj.Bz ~= 0.0)
                obj.mode.magnet = true;
                mag(1,:,:,:) = obj.Bx;
                mag(3,:,:,:) = obj.Bz;
            end

            % Calculate Energy Density
            ener = ener/(obj.fluidDetails(1).gamma - 1) ...
            + 0.5*squish(sum(mom.*mom,1))./mass...        
            + 0.5*squish(sum(mag.*mag,1));

            fluids = obj.rhoMomEtotToFluid(mass, mom, ener);
	    obj.fluidDetails.minMass = .0001*obj.rho0;
        end
    end%PROTECTED
        
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
