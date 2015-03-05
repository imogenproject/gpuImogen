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
        EOS_ISOTHERMAL = 1;
        EOS_ISOCHORIC  = 2;
        EOS_ADIABATIC  = 3;

        PROB_FIX = 1; % If problemResponse equals this, extrapolates a reasonable value instead
        PROB_ERROR = 2; % If problemResponse equals _this_, throws an error().
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
    end %PUBLIC

%===================================================================================================
    properties (Dependent = true) %                                            D E P E N D E N T [P]
        edgeFraction;
        omega0;

        rho0;
        P0;

        eqnOfState;
        omegaCurve; % = @(r) w(r) 
    end %DEPENDENT
    
%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
        pEdgeFraction;
        pOmega0;

        pRho0;
        pPress0;

        pEqnOfState;
        pOmegaCurve; % = @(r) w(r) 
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
            obj.P0                  = 1;
            obj.minMass             = 1e-5;

            obj.eqnOfState          = obj.EOS_ISOTHERMAL;
            obj.omegaCurve          = @(r) obj.omega0*(1-cos(2*pi*r));

            obj.operateOnInput(input, [64 64 1]);
        end

    function set.edgeFraction(self, f)
%        fmin = 5/self.grid(1); 
% FIXME: Not clear what to do because this can be called before grid resolution is set...
fmin = 0;
        if f <= fmin;
            warning('Positive amount of edge required: Set to 5 cells');
            f = fmin;            
        end
        self.pEdgeFraction = f;
    end
    function f = get.edgeFraction(self); f = self.pEdgeFraction; end
   
    % any w0 is acceptable;
    function set.omega0(self, w)
        self.pOmega0 = w;
    end
    function w = get.omega0(self); w = self.pOmega0; end

    function set.rho0(self, rho)
        rho = max(rho, self.minMass);
        self.pRho0 = rho;
    end
    function r = get.rho0(self); r = self.pRho0; end;

    function set.P0(self, P)
        if P <= 0; error('Pressure cannot be nonpositive!'); end

        self.pPress0 = P;
    end
    function P = get.P0(self); P = self.pPress0; end


    function set.eqnOfState(self, eos)
        if (eos < 1) || (eos > 3); error('Invalid EOS: use .EOS_ISOTHERMAL, .EOS_ISOCHORIC or .EOS_ADIABATIC'); end

        self.pEqnOfState = eos;
    end
    function e = get.eqnOfState(self); e = self.pEqnOfState; end

    function set.omegaCurve(self, f)
        self.pOmegaCurve = f;
    end
    function f = get.omegaCurve(self); f = self.pOmegaCurve; end


    end%GET/SET
    
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
    
    function polyK(self, k)
        if k <= 0; warning('P = k rho^gamma, k cannot be <= 0; Defaulted to 1'); k = 1; end
        if self.pEqnOfState ~= self.EOS_ADIABATIC; warning('Setting adiabatic K but EoS is not adiabatic.'); end        

        self.pPress0 = k*self.pRho0^self.gamma;
    end

    function cs0(self, c)
        if c <= 0; warning('cs0 cannot be nonpositive. Defaulting to 1.'); end
        if self.pEqnOfState ~= self.EOS_ISOTHERMAL; warning('Setting isothermal soundspeed but EoS is not isothermal.'); end

        self.pPress0 = self.pRho0*c*c;
    end
    end%PUBLIC
    
%===================================================================================================    
    methods (Access = protected) %                                          P R O T E C T E D    [M]                
        
%___________________________________________________________________________________________________ calculateInitialConditions
        function [mass, mom, ener, mag, statics, potentialField, selfGravity] = calculateInitialConditions(obj)

            obj.frameRotateCenter = [obj.grid(1) obj.grid(2)]/2 + .5;

            GIS = GlobalIndexSemantics();
            mygrid = GIS.pMySize;

            obj.dGrid = (1+obj.pEdgeFraction)*2./obj.grid;
 
            mom     = zeros([3 mygrid]);
            
            % Normalizes the box to radius of 1 + a bit
            [Xv Yv] = GIS.ndgridSetXY();
            Xv = (Xv - obj.grid(1)/2 + .5)*obj.dGrid(1);
            Yv = (Yv - obj.grid(2)/2 + .5)*obj.dGrid(2);

            % Evaluate the \int r w(r)^2 dr curve 
            rads   = [0:.0001:1];
            igrand = @(r) r.*obj.pOmegaCurve(r).^2;
            Rphi = fim(rads, igrand); Rphi = Rphi - Rphi(end); %Reset potential to zero at outer edge
            Rphi(end+1) = 0;

            % The analytic solution of 2d rotating-on-cylinders flow for isothermal conditions
            % for rotation curve w = w0 (1 - cos(2 pi r)). woa = pOmega0 / a^2
%            Rphi = @(x, w0) (w0^2*(15 - 24*pi^2 + 24*pi^2*x.^2 - 16*cos(2*pi*x) + cos(4*pi*x) - 32*pi*x.*sin(2*pi*x) + 4*pi*x.*sin(4*pi*x)))/(32*pi^2);

            % Isothermal density resulting from centrifugal potential
            [rho Pgas] = obj.thermo(Rphi);

            rho(10001:10002) = obj.pRho0;
            rads = [rads 1.5*(1+obj.pEdgeFraction)];
            momphi = rho .* rads .* obj.pOmegaCurve(rads);

            % Tack on static boundary region
            momphi(10001:10002) = 0;

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

        % rho(r) = pRho0 exp(-rphi / a^2)
        if obj.pEqnOfState == obj.EOS_ISOTHERMAL
            csq = obj.pPress0 / obj.pRho0;

            rho  = obj.pRho0 * exp(rphi / csq);
            Pgas = csq*rho;
        end

        % rho(r) = (rho0^gm1 + gm1 Phi / k gamma)^(1/gm1)
        if obj.pEqnOfState == obj.EOS_ADIABATIC
            k = obj.pPress0 / obj.pRho0^obj.gamma;
            gm1 = obj.gamma - 1;

            rho = (gm1*rphi/(obj.gamma*k) + obj.pRho0^gm1).^(1/gm1);
            Pgas = k * rho.^obj.gamma;
        end

        if obj.pEqnOfState == obj.EOS_ISOCHORIC
            % Reset to phi = 0 at center
            rphi = rphi - rphi(1);

            rho = ones(size(rphi))*obj.pRho0;
            Pgas = obj.pPress0 + obj.pRho0 .* rphi;
        end

    end
        
    end%PROTECTED
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]

        function help
            disp('The centrifuge initializer evolves an initial state such that a fluid rotating on cylinders with an arbitrary angular speed w(r) is in equilibrium. This is useful in diagnosing accuracy because this equilibrium is never stable and the longer it holds the better.');
            disp('');
            disp('Key simulation parameters include the equation of state (.eqnOfState = EOS_ISOTHERMAL, EOS_ADIABATIC or EOS_ISOCHORIC) and the constraints .rho0 and .P0. Adiabatic and isothermal simulations specify P and rho at the outer edge (normalized to r = 1), isochoric specifies the uniform density and the central pressure. The .gamma parameter is important too, as regardless of the initial EoS the fluid is evolved adiabatically.');
            disp('');
            disp('Key numeric parameters include .edgeFraction [2(1+edgeFraction) = resolution] and .frameRotateOmega (this simulation''s original purpose was to test the implementation of a rotating frame!)');
        end

    end
end%CLASS
