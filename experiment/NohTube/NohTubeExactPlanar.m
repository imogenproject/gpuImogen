classdef NohTubeExactPlanar < handle
    % Class annotation template for creating new classes.
    %___________________________________________________________________________________________________
    
    %===================================================================================================
    properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        xpos, xini;

        gamma;
        
        rho0, P0, m;
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
        function self = NohTubeExactPlanar(rho0, P0, m)
            self.rho0 = rho0;
            self.P0   = P0;
            self.pMach = m;

            self.gamma = 5/3;
        end

        function [rho v P] = solve(self, ignoredSpacedim, R, r0);
            
            c0 = sqrt(self.gamma*self.P0/self.rho0);
            vsh0 = self.pMach * c0;

	    d0 = size(R);
            rho = zeros(d0); v = zeros(d0); P = zeros(d0);

	    hplane = sign(R);
	    R      = abs(R);

            % Calculate how long until the 1st shock hits the wall
	    
            if r0 < 0 % 1st shock has not hit wall
	            r0 = -r0; % correct sign bit

                a = (R <  r0);
                b = (R >= r0);

                phi = (self.gamma-1+2/self.pMach^2)/(self.gamma+1);

                rho(a) = self.rho0;
                rho(b) = self.rho0/phi;
        
                v(a) = 0;
                v(b) = (self.pMach*c0*(phi-1)) .* hplane(b);

                P(a) = self.P0;
                P(b) = self.P0 * (self.gamma*(2*self.pMach^2 - 1)+1)/(self.gamma+1);
            else
                phi = (self.gamma-1+2/self.pMach^2)/(self.gamma+1);

                % calculate first postshock parameters as new preshock params
                rho1 = self.rho0/phi;
                vin  = self.pMach*c0*(phi-1);
                P1   = self.P0 * (self.gamma*(2*self.pMach^2 - 1)+1)/(self.gamma+1);

                g  = self.gamma;
                gp1 = self.gamma+1;
                gm1 = self.gamma-1;

                % Now solve the Noh tube
                c = sqrt(self.gamma*P1/rho1);
                m = vin/c;
                rho2 = rho1*(4+m*m*gp1-m*sqrt(16+m*m*gp1*gp1))/(4+2*m*m*gm1);
                Vsh  = c*((3-g)*m + sqrt(16 + gp1*gp1*m*m))/4;
                P2   = P1 + rho1*m*c^2*(m*gp1 - sqrt(16 + gp1*gp1*m*m))/4;

                self.r0 = Vsh * t0;

                a = (R <  r0);
                b = (R >= r0);

                rho(a) = rho2;
		rho(b) = rho1;

                v(a) = 0.0;
		v(b) = vin.*hplane(b);

                P(a) = P2;
		P(b) = P1;

            end

            % Remember that cell values are integral volume averages:
            % Average the cell the shock traverses
            % FIXME implement this

        end
    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                      P R O T E C T E D    [M]
    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %                                                 S T A T I C    [M]
    end%PROTECTED
    
end%CLASS
