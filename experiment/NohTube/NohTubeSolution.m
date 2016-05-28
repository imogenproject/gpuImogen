classdef NohTubeSolution < handle
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
        function self = NohTubeSolution(rho0, P0, m)
            self.rho0 = rho0;
            self.P0   = P0;
            self.pMach = m;

            self.gamma = 5/3;
        end

        function [rho v P] = solve(self, t);
            
            c0 = sqrt(self.gamma*self.P0/self.rho0);
            vsh0 = self.pMach * c0;

            d0 = size(self.xpos);
            rho = zeros(d0); v = zeros(d0); P = zeros(d0);

            % Calculate how long until the 1st shock hits the wall
            tImpact = self.xini / vsh0;

            if t < tImpact
                xShock = self.xini - t*vsh0;
                a = (self.xpos < xShock);
                b = (self.xpos >= xShock);

                phi = (self.gamma-1+2/self.pMach^2)/(self.gamma+1);

                rho(a) = self.rho0;
                rho(b) = self.rho0/phi;
        
                v(a) = 0;
                v(b) = (self.pMach*c0*(phi-1));

                P(a) = self.P0;
                P(b) = self.P0 * (self.gamma*(2*self.pMach^2 - 1)+1)/(self.gamma+1);

            else
                phi = (self.gamma-1+2/self.pMach^2)/(self.gamma+1);
                tau = t - tImpact;

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

                xShock = Vsh * tau;
                a = (self.xpos < xShock);
                b = (self.xpos >= xShock);

                rho(a) = rho2; rho(b) = rho1;
                v(a) = 0.0;    v(b) = vin;
                P(a) = P2;     P(b) = P1;

            end

            % Remember that cell values are integral volume averages:
            % Average the cell the shock traverses
            i0 = numel(find(a)); % center of cell left
            if i0 < numel(a) 
                x0 = self.xpos(i0);
                x1 = self.xpos(i0+1);
                delta = (xShock-x0)/(x1-x0);
                if delta < .5 % left cell needs avg
                    delta = delta + .5;
                    rho(i0) = delta*rho(i0) + (1-delta)*rho(i0+1);
                    v(i0)   = delta*v(i0)   + (1-delta)*v(i0+1);
                    P(i0)   = delta*P(i0)   + (1-delta)*P(i0+1);
                else % rite cell needs avg
                    delta = delta - .5;
                    rho(i0+1) = delta*rho(i0) + (1-delta)*rho(i0+1);
                    v(i0+1)   = delta*v(i0)   + (1-delta)*v(i0+1);
                    P(i0+1)   = delta*P(i0)   + (1-delta)*P(i0+1);
                end
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
