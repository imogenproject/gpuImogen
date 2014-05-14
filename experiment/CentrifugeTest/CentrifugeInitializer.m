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
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        edgeFraction;
        omega0;
        rho0;
        a_isothermal;

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
            obj.a_isothermal        = 1;
            obj.minMass             = 1e-5;

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

%            if (obj.grid(3) > 1)
%                if obj.useZMirror == 1
%                    obj.bcMode.z    = ENUM.BCMODE_FLIP;
%                else
%                    obj.bcMode.z    = ENUM.BCMODE_FADE; 
%                end
%            else
%                obj.bcMode.z    = ENUM.BCMODE_CONST;
%            end

            obj.frameRotateOmega = obj.omega0;
            obj.frameRotateCenter = [obj.grid(1) obj.grid(2)]/2 + .5;

            GIS = GlobalIndexSemantics();
            mygrid = GIS.pMySize;

            obj.dGrid = (1+obj.edgeFraction)*2./obj.grid;
 
            mom     = zeros([3 mygrid]);
            
            [Xv Yv] = GIS.ndgridSetXY();
            Xv = (Xv - obj.grid(1)/2 + .5)*obj.dGrid(1);
            Yv = (Yv - obj.grid(2)/2 + .5)*obj.dGrid(2);

            % Normalizes the box to radius of 1 + a bit

            % The analytic solution of 2d rotating-on-cylinders flow for isothermal conditions
            % for rotation curve w = w0 (1 - cos(2 pi r)). woa = omega0 / a^2
            rhoAnalytic = @(r, woa) exp( -(-1 + 2*pi*pi*(1-r.^2) + cos(2*pi*r) + 2*pi*r.*sin(2*pi*r))*woa/(4*pi*pi));
            % p_phi = rho r omega
            momPhiAnalytic = @(r, w0, a) rhoAnalytic(r, w0*a^-2) .* r .* obj.omega0 .* (1 - cos(2*pi*r) - 1);
            rads   = [0:.0001:1.0002 1.5*(1+obj.edgeFraction)];
            rhos   = obj.rho0 * rhoAnalytic(rads, obj.omega0 *obj.a_isothermal^-2);
            momphi = momPhiAnalytic(rads, obj.omega0, obj.a_isothermal);

            rhos(10001:end) = obj.rho0;
            momphi(10001:end) = -obj.rho0*rads(10001:end)*obj.omega0;

            % FIXME: This will take a dump if Nz > 1...
            gridR = sqrt(Xv.^2+Yv.^2);
            mass(:,:,1) = interp1(rads, rhos, gridR);
            mass(gridR > 1.0) = obj.rho0;
            
            mom(1,:,:,1) = -Yv .* interp1(rads, momphi, gridR) ./ gridR;
            mom(2,:,:,1) = Xv  .* interp1(rads, momphi, gridR) ./ gridR;

            mass    = max(mass, obj.minMass);
            mag     = zeros([3 mygrid]);
            
            ener    = obj.a_isothermal^2 * mass / (obj.gamma-1)...
                        + 0.5*squeeze(sum(mom .* mom, 1)) ./ mass ...           % kinetic energy
                        + 0.5*squeeze(sum(mag .* mag, 1));                      % magnetic energy                    
            
            statics = [];%StaticsInitializer(obj.grid);

            selfGravity = [];
            potentialField = [];%PotentialFieldInitializer();

        end
        
    end%PROTECTED
        
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
