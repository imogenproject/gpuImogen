classdef CentrifugeInitializer < Initializer
% Uses equilibrium routines to create an equilibrium disk configuration and then formats them for
% use in Imogen based on the Kojima model. The disk is created in the polytropic unit systems under
% the constraints that the gravitational constant, G, the mass of the central star, M, and the 
% polytropic constant, K, are all unity.
%
% Unique properties for this initializer:
%   q                 angular velocity exponent (omega = omega_0 * r^-q.              double
%   radiusRatio       (inner disk radius) / (radius of density max).                  double
%   edgePadding       number of cells around X & Y edges to leave blank.              double
%   pointRadius       size of the softened point at the center of the grid            double
%   useStatics        specifies if static conditions should be set for the run.       logical
    
        
%===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]
        EOS_ISOTHERM = 1;
        EOS_ISODENSITY = 2;
        EOS_ADIABATIC = 3;
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        edgeFraction;
        omega0;

        rho0;

        eqnOfState;
        omegaCurve; % = @(r) w(r) 

        polyK;
        cs0;

    end %PUBLIC

%===================================================================================================
    properties (Dependent = true) %                                            D E P E N D E N T [P]
    end %DEPENDENT
    
%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
    end %PROTECTED
    
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        
%___________________________________________________________________________________________________ KojimaDiskInitializer
        function obj = CentrifugeInitializer(input)
            obj                     = obj@Initializer();            
            obj.runCode             = 'CENTRIFUGE';
            obj.info                = 'Code test for rotating-frame source term.';
            obj.mode.fluid          = true;
            obj.mode.magnet         = false;
            obj.mode.gravity        = true;
            obj.iterMax             = 300;
            obj.bcMode.x            = ENUM.BCMODE_CONST;
            obj.bcMode.y            = ENUM.BCMODE_CONST;
            obj.bcInfinity          = 5;
            obj.activeSlices.xy     = true;
            obj.timeUpdateMode      = ENUM.TIMEUPDATE_PER_STEP;
            
            obj.edgeFraction        = .1;
            obj.omega0              = .5;
            obj.rho0                = 1;
            obj.cs0                 = 1;
            obj.polyK               = 1;
            obj.minMass             = 1e-5;

            obj.eqnOfState          = obj.EOS_ISOTHERM;
            obj.omegaCurve          = @(r) obj.omega0*(1-cos(2*pi*r));

            obj.operateOnInput(input, [64 64 1]);

        end
        
%___________________________________________________________________________________________________ GS: pointMass
% Dynmaic pointMass property to connect between the gravity vars structure and run files.
    
    end%GET/SET
    
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
    end%PUBLIC
    
%===================================================================================================    
    methods (Access = protected) %                                          P R O T E C T E D    [M]                
        
%___________________________________________________________________________________________________ calculateInitialConditions
        function [mass, mom, ener, mag, statics, potentialField, selfGravity] = calculateInitialConditions(obj)

            obj.frameRotateCenter = [obj.grid(1) obj.grid(2)]/2 + .5;

            GIS = GlobalIndexSemantics();
            mygrid = GIS.pMySize;

            obj.dGrid = (1+obj.edgeFraction)*2./obj.grid;
 
            mom     = zeros([3 mygrid]);
            
            % Normalizes the box to radius of 1 + a bit
            [Xv Yv] = GIS.ndgridSetXY();
            Xv = (Xv - obj.grid(1)/2 + .5)*obj.dGrid(1);
            Yv = (Yv - obj.grid(2)/2 + .5)*obj.dGrid(2);

            % Evaluate the \int r w(r)^2 dr curve 
            rads   = [0:.0001:1];
            igrand = @(r) r.*obj.omegaCurve(r).^2;
            Rphi = fim(rads, igrand); Rphi = Rphi - Rphi(end); %Reset potential to zero at outer edge
            Rphi(end+1) = 0;

            % The analytic solution of 2d rotating-on-cylinders flow for isothermal conditions
            % for rotation curve w = w0 (1 - cos(2 pi r)). woa = omega0 / a^2
%            Rphi = @(x, w0) (w0^2*(15 - 24*pi^2 + 24*pi^2*x.^2 - 16*cos(2*pi*x) + cos(4*pi*x) - 32*pi*x.*sin(2*pi*x) + 4*pi*x.*sin(4*pi*x)))/(32*pi^2);

            % Isothermal density resulting from centrifugal potential
            [rho Pgas] = obj.thermo(Rphi);

            rho(10001:10002) = obj.rho0;
            rads = [rads 1.5*(1+obj.edgeFraction)];
            momphi = rho .* rads .* (obj.omegaCurve(rads) - obj.frameRotateOmega);

            % Tack on static boundary region
            momphi(10001:10002) = -obj.rho0*rads(10001:10002)*obj.frameRotateOmega;

            % FIXME: This will take a dump if Nz > 1...
            gridR = sqrt(Xv.^2+Yv.^2);
            mass(:,:,1) = interp1(rads, rho, gridR);
            
            mom(1,:,:,1) = -Yv .* interp1(rads, momphi, gridR) ./ gridR;
            mom(2,:,:,1) = Xv  .* interp1(rads, momphi, gridR) ./ gridR;

            mass    = max(mass, obj.minMass);
            mag     = zeros([3 mygrid]);
            
            pressure = interp1(rads, Pgas, gridR);

            ener    = pressure / (obj.gamma-1) ...
                        + 0.5*squeeze(sum(mom .* mom, 1)) ./ mass ...           % kinetic energy
                        + 0.5*squeeze(sum(mag .* mag, 1));                      % magnetic energy                    
            
            statics = [];%StaticsInitializer(obj.grid);

            selfGravity = [];
            potentialField = [];%PotentialFieldInitializer();

        end
    % Given the centrifuge potential, integral(r w(r)^2), 
    % Solves the integral(P' / rho) side 
    function [rho, Pgas] = thermo(obj, rphi)

	% rho(r) = rho0 exp(-rphi / a^2)
        if obj.eqnOfState == obj.EOS_ISOTHERM
            rho  = obj.rho0 * exp(rphi / obj.cs0^2);
            Pgas = obj.cs0^2*rho;
	end

	% rho(r) = 
	if obj.eqnOfState == obj.EOS_ADIABATIC
            gm1 = obj.gamma - 1;
	    rho = (gm1*rphi/(obj.gamma*obj.polyK) + obj.rho0^gm1).^(1/gm1);
            Pgas = obj.polyK * rho.^obj.gamma;
	end

	% Danger Zone: This has regions of no-solution!
	if obj.eqnOfState == obj.EOS_ISODENSITY
            rho = ones(size(rphi))*obj.rho0;
            Pgas = obj.cs0^2 * obj.rho0 * obj.gamma - obj.rho0 * rphi;

            if any(Pgas < 0);
                error('Fatal: Centrifuge solution thermodynamically impossible.');
            end
	end

    end
        
    end%PROTECTED
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
