classdef NohTubeColdGeneral < handle
    % Class annotation template for creating new classes.
    %___________________________________________________________________________________________________
    
    %===================================================================================================
    properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        gamma;
        
        rho0, P0, m;
        r0;

	v0;
    end %PUBLIC
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
        pMach;
    end %PROTECTED
    
    %===================================================================================================
    methods %                                                                     G E T / S E T  [M]
    end%GET/SET
    
    %===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
        function self = NohTubeColdGeneral(rho0, P0, m)
            self.rho0 = rho0;
            self.P0   = P0;
            self.pMach = m;

            self.gamma = 5/3;

	    c0 = sqrt(self.gamma*self.P0/self.rho0);
	    self.v0 = -c0*self.pMach;
        end

        function [rho, v, P] = solve(self, spaceDim, R, r0)
            % Solves the N-dimensional radial strong shock equations
            % Works in D > 1 by neglecting pressure term
            % R : Array of radii (must be positive semidefinite)
            % t0: Sign determines if implosion (t0 < 0) or explosion (t0 > 0)

            Dee = (1-self.gamma)*self.v0/2;
            t = r0 / Dee;
	    r0 = abs(r0);

            rho = zeros(size(R));
            v   = zeros(size(R));
            P   = zeros(size(R));

            rho(R > r0) = self.rho0*(1 - self.v0*abs(t)./R(R > r0)).^(spaceDim-1);
            v(R > r0)    = self.v0;

            % KE plus small qty for finite pressure requirement, defined thru Mach
            % Define density at shock
            jc = (self.gamma+1)/(self.gamma-1);
            rhopre = self.rho0*jc^(spaceDim-1);
            % The pressure for the Mach to be such
            p0 = rhopre*(self.v0/self.pMach)^2/(self.gamma*(self.gamma-1));

            % Set unshocked pressure assuming adiabatic flow
            P(R > r0) = p0*(rho(R > r0)./rhopre).^self.gamma;

            if t > 0
                % Stationary doubly-shocked central region
                mcent = self.rho0 * jc^spaceDim;
                rho(R <= r0)= mcent;
                % mom is zero by default: yay
                P(R <= r0)= (self.gamma-1)*mcent*self.v0^2 / 2;
            else
	        % Shell imploding a vaccum
                rho(R <= r0) = 0;
                P(R <= r0) = 0;
            end

        end
    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                      P R O T E C T E D    [M]
    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %                                                 S T A T I C    [M]
    end%PROTECTED
    
end%CLASS
