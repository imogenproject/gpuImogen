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
        
        function [rho, v, P] = solve(self, ignoredSpacedim, R, r0)
            % [rho, v, P] = solve(self, ignoredSpacedim, R, r0)
            % rho < density
            % v   < velocity
            % P   < pressure
            % R   > Points to evaluate solution at 
            % r0  > Shock position; r0 < 0 means first shock has not hit wall
            %                       r0 > 0 means second shock is rebounding
            % This function returns the exact solution of a Noh tube for any fluid conditions
            % and 1st shock strength evaluated at points R when the shock position is r0.
            
            c0 = sqrt(self.gamma*self.P0/self.rho0);

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

                gp1 = self.gamma+1;
                gm1 = self.gamma-1;

                % Solve RP(preshock) again yielding the double-shocked values
                c = sqrt(self.gamma*P1/rho1);
                m0 = vin/c;
                msq = m0^2;
                rho2 = rho1*(4+msq*gp1-m0*sqrt(16+msq*gp1*gp1))/(4+2*msq*gm1);
                %Vsh  = c*((3-g)*m0 + sqrt(16 + gp1*gp1*msq))/4;
                P2   = P1 + rho1*m0*c^2*(m0*gp1 - sqrt(16 + gp1*gp1*msq))/4;

                %self.r0 = Vsh * t0;

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
        
        function r = shockPositionGivenTime(self, r0, t)
            % r = shockPositionGivenT(r0, t) returns the position of
            % the shock at time t, when the position at t=0 is r0.
            % r0 < 0 -> prior to wall rebound
            
            c0 = sqrt(self.gamma*self.P0/self.rho0);
            vsh0 = self.pMach * c0;
            
            if r0 < 0 % shock has not hit wall yet
                tHit = (-r0)/vsh0; % when it will hit
                if t < tHit
                    r = r0 + vsh0 * t; % this is still < 0 because it hasn't hit yet
                else
                    t2 = t - tHit; % time elapsed since wall rebound
                    vsh1 = self.computeSecondShockSpeed();
                    r = t2*vsh1;
                end                
            else % r(t=0) has already rebounded:
                vsh1 = self.computeSecondShockSpeed();
                r = r0 + vsh1*t;
            end
           
        end
        
        function t = shockTimeGivenPosition(self, r0, r)
            % t = shockTimeGivenPosition(r0, r)
            % computes the time required for the shock to move from
            % position r0 to position r
            
            vsh0 = self.computeFirstShockSpeed();
            vsh1 = self.computeSecondShockSpeed();
            
            if r > r0 % t is positive
                if r0 < 0 % start before 1st impact
                    if r > 0 % end after it
                        tHit = -r0 / vsh0;
                        tRebound = r / vsh1;
                        t = tHit + tRebound;
                    else % end before it
                        t = (r0 - r)/vsh0;
                    end
                else % start after impact
                    t = (r - r0)/vsh0;
                end
            else % t is negative or zero
                % take advantage of time reversal
               t = -self.shockTimeGivenPosition(r, r0);
            end
                
        end
        
        function vsh0 = computeFirstShockSpeed(self)
            c0 = sqrt(self.gamma*self.P0/self.rho0);
            vsh0 = self.pMach * c0;
        end
        
        function vsh1 = computeSecondShockSpeed(self)
            c0 = sqrt(self.gamma*self.P0/self.rho0);
            phi = (self.gamma-1+2/self.pMach^2)/(self.gamma+1);
            rho1 = self.rho0/phi;
            P1   = self.P0 * (self.gamma*(2*self.pMach^2 - 1)+1)/(self.gamma+1);
            gp1 = self.gamma+1;
            gm1 = self.gamma-1;
            
            vin  = self.pMach*c0*(phi-1);
            % Solve RP(preshock) again yielding the double-shocked values
            c = sqrt(self.gamma*P1/rho1);
            m0 = vin/c;
            msq = m0^2;
            %rho2 = rho1*(4+msq*gp1-m0*sqrt(16+msq*gp1*gp1))/(4+2*msq*gm1);
            vsh1  = c*((3-self.gamma)*m0 + sqrt(16 + gp1*gp1*msq))/4;
            
        end
    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                      P R O T E C T E D    [M]
    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %                                                 S T A T I C    [M]
    end%PROTECTED
    
end%CLASS
