classdef SedovSolver < handle
    % The SedovSolver class is simply a dumping ground to collect together a few of the functions
    % involved in generating reference solutions to the Sedov-Taylor problem for Imogen
    % The implementation is per the following report from LANL:
    % ----------------
    % Kamm, James R., and F. X. Timmes. On efficient generation of numerically robust Sedov
    % solutions. Technical Report LA-UR-07-2849, Los Alamos National Laboratory, 2007.
    % ----------------
    % While a closely related writeup by Kamm 2000 provides a step-by-step description
    %___________________________________________________________________________________________________
    
    %===================================================================================================
    properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
    end %PUBLIC
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
    end %PROTECTED
    
    %===================================================================================================
    methods %                                                                     G E T / S E T  [M]
    end%GET/SET
    
    %===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                      P R O T E C T E D    [M]
    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %                                                 S T A T I C    [M]
        
        function alpha = findAlpha(E, rho0, gamma, j)
            % alpha = findAlpha(E, rho0, gamma, j) computes the numerical integrals giving the scaling
            % factor alpha in the ST solution

            % The code from KT07 applies to all 0 <= w <= j but we simply set w to 0.
            w=0;
            j2w = j+2-w;

            % Kamm 2000 eqn 18/19, KT07 eqns 5/6; KT07 eqn 5 is wrong but the demo code is correct;
            % Kamm eqn 18 is correct
            v2 = 4/(j2w*(gamma+1));
            vstar = 2/((gamma-1)*j+2);

            % This code only evaluates the standard case

            % KT07 25-29, Kamm 33-37
            a = j2w*(gamma+1)/4;
            b = (gamma+1)/(gamma-1);
            c = gamma*.5*j2w;
            d = j2w*(gamma+1)/(j2w*(gamma+1)-2*(2+j*(gamma-1)));
            e = 1+j*(gamma-1)/2;

            % KT07 19-24, Kamm 42-47
            alpha0 = 2/j2w;
            alpha2 = (1-gamma)/(2*(gamma-1)+j-gamma*w);
            alpha1 = ((j2w*gamma)/(2+j*(gamma-1)))*(2*(j*(2-gamma)-w)/(gamma*j2w^2) -alpha2 );
            alpha3 = (j-w)/(2*(gamma-1)+j-gamma*w);
            alpha4 = j2w*(j-w)*alpha1/(j*(2-gamma)-w);
            alpha5 = (w*(1+gamma)-2*j)/(j*(2-gamma)-w);

            % Evaluation of energy integrals
            % KT07 eqns 35, 55, 56
            % J1 (Kamm eqn 67 / 73)
            J1_Integrand = @(v) -((gamma+1)/(gamma-1)) * v^2 * (alpha0/v + alpha2*c/(c*v-1) - alpha1*e/(1-e*v)) * ( (a*v)^alpha0 * (b*(c*v-1))^alpha2 * (d*(1-e*v))^alpha1)^-j2w * (b*(c*v-1))^alpha3 * (d*(1-e*v))^alpha4 * (b*(1-c*v/gamma))^alpha5;
            J1A = @(v) arrayfun(J1_Integrand, v);

            % J2 (Kamm eq eq 68 / 74)
            J2_Integrand = @(v) -.5*(1+1/gamma)*v^2*((c*v-gamma)/(1-c*v))*(alpha0/v + alpha2*c/(c*v-1) - alpha1*e/(1-e*v)) * ( (a*v)^alpha0*(b*(c*v-1))^alpha2 *(d*(1-e*v))^alpha1 )^-j2w * (b*(c*v-1))^alpha3*(d*(1-e*v))^alpha4 * (b*(1-c*v/gamma))^alpha5;
            J2A = @(v) arrayfun(J2_Integrand, v);

            % Lower integral bound for standard case, KT07 eq 61, Kamm eqn 23
            vmin =  2/(j2w*gamma);

            J1 = integral(J1A, vmin, v2);
            J2 = integral(J2A, vmin, v2);

            % KT07 eqns 58-60, Kamm eqns 57/58/66
            I1 = 2^(j-2)*(1*(j==1) + pi*(j==2) + pi*(j==3))*J1;
            I2 = 2^(j-1)*(1*(j==1) + pi*(j==2) + pi*(j==3))*J2/(gamma-1);
            alpha = I1+I2;
        end

        function [rho vradial p] = FlowSolution(E, t, radii, rho0, gamma, j)
            % [rho V p] = SolutionGenerator(E, t, radii, rho0, gamma, j) computes the exact solution
            % of the Sedov-Taylor explosion (see Kamm & Timmes 2007) of energy E at time t > 0 at
            % the radial points given by 'radii' with preshock fluid density rho0, polytropic index 
            % gamma, and j-dimensional spatial symmetry (1=planar, 2=cylindrical, 3=spherical) in a 
            % polytropic fluid of index gamma.

            % The code from KT07 applies to all 0 <= w <= j but we simply set w to 0.
            w=0;
            j2w = j+2-w;
            
            % Kamm 2000 eqn 18/19, KT07 eqns 5/6; KT07 eqn 5 is wrong but the demo code is correct;
            % Kamm eqn 18 is correct
            v2 = 4/(j2w*(gamma+1));
            vstar = 2/((gamma-1)*j+2);
            
            % This code only evaluates the standard case

            % KT07 25-29, Kamm 33-37
            a = j2w*(gamma+1)/4;
            b = (gamma+1)/(gamma-1);
            c = gamma*.5*j2w;
            d = j2w*(gamma+1)/(j2w*(gamma+1)-2*(2+j*(gamma-1)));
            e = 1+j*(gamma-1)/2;
            
            % KT07 19-24, Kamm 42-47
            alpha0 = 2/j2w;
            alpha2 = (1-gamma)/(2*(gamma-1)+j-gamma*w);
            alpha1 = ((j2w*gamma)/(2+j*(gamma-1)))*(2*(j*(2-gamma)-w)/(gamma*j2w^2) -alpha2 );
            alpha3 = (j-w)/(2*(gamma-1)+j-gamma*w);
            alpha4 = j2w*(j-w)*alpha1/(j*(2-gamma)-w);
            alpha5 = (w*(1+gamma)-2*j)/(j*(2-gamma)-w);
            
            % Lower integral bound for standard case, KT07 eq 61, Kamm eqn 23
            vmin =  2/(j2w*gamma);
            
            % We shovel this off into a separate function to reduce code duplication
            % even though here virtually all the preceeding definitions are needed anyway
            alpha = SedovSolver.findAlpha(E, rho0, gamma, j);
            
            % shock position (eq 14)
            r2 = (E*t^2/(rho0*alpha))^(1/j2w);
            
            % shock speed (eq 16)
            U = 2*r2/(t*j2w);
            
            % preshock density (eq 5)
            rho1 = rho0*r2^(-w);
            
            % Instantaneous postshock state (eq 13) from RH conditions
            vpost   = 2*U/(gamma+1);
            rho2 = (gamma+1)*rho1/(gamma-1);
            p2   = 2*rho1*U^2 / (gamma+1);
            
            % Split into pre- and post-shock sets
            split = (radii < r2);
            
            rShocked   = radii(split);
            rUntouched = radii(~split);
            
            % ASSUME STANDARD CASE: eqns 38-41
            % Calculate the state of the parts that have gotten explodey
            
            % First solve r = r2 lambda(v) (KT07 eqn 34, Kamm eq 38) for v since we're given r
            lambda = @(v) (a*v).^-alpha0 .* (b*(c*v-1)).^-alpha2 .* (d*(1-e*v)).^-alpha1;
            vset = arrayfun(@(l0) fzero( @(x) lambda(x)*r2 - l0, [vmin v2]), rShocked);
            
            % KT eqn 30-33, Kamm eqn 29-32: facilitate evaluation of output state
            x1 = a*vset;
            x2 = b*(c*vset-1);
            x3 = d*(1-e*vset);
            x4 = b*(1-c*vset/gamma);
            
            % KT07 eqn 36-38, Kamm 39-41
            vboom = vpost * lambda(vset); 
            rhoboom = rho2 * x1.^(w*alpha0) .* x2.^(alpha3+alpha2*w) .* x3.^(alpha4+alpha1*w) .* x4.^(alpha5);
            Pboom = p2     * x1.^(j*alpha0) .* x3.^(alpha4+alpha1*(w-2)) .* x4.^(alpha5+1);
            
            rho = rho0*ones(size(radii)); rho(split) = rhoboom;
            vradial = zeros(size(radii)); vradial(split) = vboom;
            p = zeros(size(radii));       p(split) = Pboom;
            
        end
        
        function t = timeUntilSize(E, R, rho0, gamma, j)
            % tSize = Sedov_timeToReachSize(E, R, rho0, gamma, j)
            % exact solution of the Sedov-Taylor explosion (see Kamm & Timmes 2007)
            % at the r points given by radii in the j-dimension symmetry (1=plane,
            % 2=cylindrical, 3=spherical) in a polytropic gas of index gamma

            alpha = SedovSolver.findAlpha(E, rho0, gamma, j);
            % shock position (eq 14) solved for t given r
            w=0;
            j2w = j+2-w;
            t = sqrt(rho0*alpha*R^j2w / E);
        end
        
        function [R, Q] = RadialMap(array,center,nSamples)
            % [R, Q] = RadialMap(array, center, nSamples) finds the radius
            % in cells of every point in array from the coordinate-index
            % center, then randomly picks nSamples values and returns
            % [R(sample_indexes) array(sample_indexes)]
            
            if (nargin < 1 || isempty(array)), error('Imogen:DataInputError','No array specified. Operation aborted.'); end
            nDim = ndims(array);
            grid = size(array);
            
            if (nargin < 3) || isempty(nSamples); nSamples = 10000; end
            if (nargin < 2 || isempty(center)), center = round(grid / 2.0); end
            
            if (nDim == 2)
                [xg yg] = ndgrid((1:grid(1))-center(1), (1:grid(2))-center(2));
                R = sqrt(xg.^2+yg.^2);
            else
                [xg yg zg] = ndgrid((1:grid(1))-center(1), (1:grid(2))-center(2), (1:grid(3))-center(3));
                R = sqrt(xg.^2+yg.^2+zg.^2);
            end
            
            % if nSamples << numel(array) results will be almost all unique
            pick = ceil((numel(array)-1)*rand(nSamples,1));
            
            R = R(pick);
            Q = array(pick);
        end
        
        
    end%PROTECTED
    
end%CLASS
