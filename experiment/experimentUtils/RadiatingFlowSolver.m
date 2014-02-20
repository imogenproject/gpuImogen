classdef RadiatingFlowSolver < handle
% This little ditty solves the equations of a one dimensional free-radiating and free-flowing
% flow. This is defined in the hydrodynamic case by:
% d/dx(rho vx)        = 0
% d/dx(rho vx vy + P) = 0
% d/dx(rho vx vy)     = 0
% d/dx(vx(T + gamma P / gamma-1)) = lambda
% where 'lambda' is an arbitrary radiation function

    properties (SetAccess = public, GetAccess = public)
        rho0, vx0, vy0, bx0, by0, Pgas0, gamma, beta, theta; % Analytic parameters
        Mt, Nt; % Pade coefficient levels

    end

    properties (SetAccess = private, GetAccess = private)
        px; % Conserved quantity, = rho0 vx0
        fx; % Conserved quantity, = rho0 vx0^2 + P0
        fy; % Conserved quantity, = rho0 vy0 (~vy0 constant)

        isHydro;
        ABHistory; % Previous values of vx' used by the AB integrator
    end

    properties (SetAccess = private, GetAccess = public)
        integErrorFlag;
        flowSolution;
    end

    methods (Access = public)
        % Class constructor
        function self = RadiatingFlowSolver(rho, vx, vy, bx, by, P, gamma, beta, theta)
% RadiatingFlowSolver class
% RFS(rho, vx, vy, bx, by, P, gamma, beta, theta) sets up to solve a radiating flow's equilibrium 
% with the given initial conditions.
            if(nargin < 9)
               disp('Require:  RadiatingFlowSolver(rho, vx, vy, bx, by, P, gamma, beta, theta)');
            end
            self.rho0 = rho;
            self.vx0 = vx;
            self.vy0 = vy;

            self.bx0 = bx;
            self.by0 = by;

            if ((bx == 0) && (by == 0)); self.isHydro = 1; else; self.isHydro = 0; end

            self.Pgas0 = P;
            self.gamma = gamma;
            self.beta = beta;
            self.theta = theta; 

            self.px = self.vx0 * self.rho0;
            self.fx = self.px*self.vx0 + self.Pgas0 - bx*bx/2;
            self.fy = rho*vx*vy-bx*by;

            self.numericalSetup(1,1)
            self.integErrorFlag = 0;
        end

        % Solvers for primitives based on conserved quantities
        function x = rho(self, vx);    x = self.px ./ vx;        return; end
        function x = by(self, vx);     x = (self.bx0 * self.fy)/(-self.bx0^2 + self.px * vx); return; end
        function x = vy(self, vx);     x = (self.fy * vx)/(-self.bx0^2 + self.px * vx); return; end
        function x = Ptherm(self, vx); x = self.fx - self.px * vx + self.bx0^2 *(1/2 - self.fy^2/(2* (self.bx0^2 - self.px * vx)^2)); return; end
        function x = Pmag(self, vx);   x = (self.bx0^2 + self.by(vx)^2)/2; return; end

        function numericalSetup(self, N, M)
            self.Nt = N;
            self.Mt = M;
        end

        function lx = coolingLength(self, v)
            if nargin == 1; % vx not given
                v = self.vx0;
            end

            einternal = self.Pthermtherm(v) / (self.gamma-1.0);
            radrate   = self.beta * self.rho(v)^(2.0-self.theta) * self.Ptherm(v)^self.theta;

            lx = v * einternal / radrate;
        end
        
        function tcool = coolingTime(self, v)
            einternal = self.Ptherm(v) / (self.gamma-1.0);
            radrate   = self.beta * self.rho(v)^(2.0-self.theta) * self.Ptherm(v)^self.theta;
            
            tcool = einternal/radrate;
        end

        function vp = calculateVprime(self, vx)

            Pgas = self.Ptherm(vx);
            PX = self.px;
            FX = self.fx;
            FY = self.fy;
            den = self.rho(vx);

            % Calculate the derivative using the same v1 formula as below
            if self.isHydro;
                vp = (Pgas^self.theta*self.beta*(-1 + self.gamma)*den^(2 -self.theta))/(PX*vx - Pgas*self.gamma);
            else
                BX = self.bx0;
                BY = self.by(vx);
                vp = (2*Pgas^self.theta*PX^2*(-BX^2 + PX*vx)^3*self.beta*(-1 + self.gamma))/ (vx^2*(BX^8*self.gamma + BX^6*(PX*vx*(-2 - 5*self.gamma) + 2*FX*self.gamma) + 2*PX^3*vx^3*(-(FX*self.gamma) + PX*vx*(1 + self.gamma)) + BX^4*(-(FY^2*self.gamma) - 6*FX*PX*vx*self.gamma + PX^2*vx^2*(6 + 9*self.gamma)) + BX^2*PX*vx*(FY^2*(-2 + self.gamma) + PX*vx*(PX*vx*(-6 - 7*self.gamma) + 6*FX*self.gamma)))*den^self.theta);
            end

        end

        % Takes steps using an explicit high-order taylor series to provide history points for the restart
        function restartAdamsBashforth(self, vx, h)

            self.ABHistory = [0 0 0 0 0];

            self.ABHistory(1) = self.calculateVprime(vx);

            for N = 2:5
                newv = self.takeTaylorStep(vx, h);
                self.ABHistory(N) = self.calculateVprime(newv);
                self.flowSolution(end+1,:) = [(self.flowSolution(end,1)+h) newv];
                vx = newv;
            end

        end

        function newv = takeTaylorStep(self, vx, h)
            Pgas = self.Ptherm(vx);
            PX = self.px;
            FX = self.fx;
            FY = self.fy;
            BX = self.bx0;
            BY = self.by(vx);
            den = self.rho(vx);
            G = self.gamma;
            gp1 = self.gamma + 1;
 
            if self.isHydro;
                v1 = (Pgas^self.theta*self.beta*(-1 + G)*den^(2 - self.theta))/(PX*vx - Pgas*G);
                v2 = -(PX*(-1 + G)*v1*((Pgas^(-1 + self.theta)*self.beta*(Pgas*(2 - self.theta) + PX*vx*self.theta)*den^(2 - self.theta))/(PX*vx) + ((1 + G)*v1)/(-1 + G)))/(2.*(PX*vx - Pgas*G));
                v3 = -((PX*(1 + G)*v1*v2)/(PX*vx - Pgas*G)) - (Pgas^(-1 + self.theta)*(-1 + G)*den^(2 - self.theta)* (-((self.beta*(6*Pgas^2 + (-FX^2 + 6*FX*Pgas - 10*Pgas^2)*self.theta + (FX - 2*Pgas)^2*self.theta^2)*v1^2)/(Pgas*vx^2)) + 2*self.beta*(-((Pgas*(-2 + self.theta))/vx) + PX*self.theta)*v2))/ (6.*(PX*vx - Pgas*G));
                v4 = -(PX*(1 + G)*(v2^2 + 2*v1*v3))/(2.*(PX*vx - Pgas*G)) - (Pgas^(-3 + self.theta)*self.beta*(-1 + G)*den^(1 - self.theta)*(FX^3*(-2 + self.theta)*(-1 + self.theta)*self.theta*den*v1^3 - 4*Pgas^3*(-1 + self.theta)*(-3 + 2*self.theta)*v1*((-2 + self.theta)*den*v1^2 + 3*PX*v2) + 12*Pgas^4*(-1 + self.theta)*v3 + FX*(6*Pgas^2*self.theta*(-3 + 2*self.theta)*v1*((-2 + self.theta)*den*v1^2 + 2*PX*v2) + 6*Pgas^3*(2 - 3*self.theta)*v3) + FX^2*(-6*Pgas*(-1 + self.theta)*self.theta*v1*((-2 + self.theta)*den*v1^2 + PX*v2) + 6*Pgas^2*self.theta*v3)))/(24.*vx^3*(PX*vx - Pgas*G));
                v5 = -((PX*(v2*v3 + G*v2*v3 + v1*v4 + G*v1*v4))/(PX*vx - Pgas*G)) + (Pgas^self.theta*self.beta*(-1 + G)*den^(3 - self.theta)*((6*vx*(-2 + self.theta)*self.theta*(PX*(-1 + self.theta)*v1^2 - 2*Pgas*v2)*((-3 + self.theta)*v1^2 + 2*vx*v2))/Pgas^2 + (4*vx^2*(-2 + self.theta)*self.theta*v1*(-(PX^2*(-2 + self.theta)*(-1 + self.theta)*v1^3) + 6*Pgas*PX*(-1 + self.theta)*v1*v2 - 6*Pgas^2*v3))/Pgas^3 - (4*(-2 + self.theta)*self.theta*v1*((12 - 7*self.theta + self.theta^2)*v1^3 + 6*vx*(-3 + self.theta)*v1*v2 + 6*vx^2*v3))/Pgas + (vx^3*self.theta*(PX^3*(-3 + self.theta)*(-2 + self.theta)*(-1 + self.theta)*v1^4 - 12*(FX - Pgas)*Pgas*(2 - 3*self.theta + self.theta^2)*den*v1^2*v2 + 12*Pgas^2*PX*(-1 + self.theta)*(v2^2 + 2*v1*v3) - 24*Pgas^3*v4))/Pgas^4 + ((-2 + self.theta)*(((-60 + 47*self.theta - 12*self.theta^2 + self.theta^3)*v1^4)/vx + 12*(12 - 7*self.theta + self.theta^2)*v1^2*v2 + 24*vx*(-3 + self.theta)*v1*v3 + 12*vx*((-3 + self.theta)*v2^2 + 2*vx*v4)))/PX))/(120.*vx^2*(PX*vx - Pgas*G));
                v6 = -(PX*(1 + G)*(v3^2 + 2*v2*v4 + 2*v1*v5))/(2.*(PX*vx - Pgas*G)) + (Pgas^self.theta*self.beta*(-1 + G)*den^(2 - self.theta)*((den*((2*vx*(-2 + self.theta)*self.theta*((-3 + self.theta)*v1^2 + 2*vx*v2)* (-(PX^2*(-2 + self.theta)*(-1 + self.theta)*v1^3) + 6*Pgas*PX*(-1 + self.theta)*v1*v2 - 6*Pgas^2*v3))/Pgas^2 + (2*(-2 + self.theta)*self.theta*(PX*(-1 + self.theta)*v1^2 - 2*Pgas*v2)*((12 - 7*self.theta + self.theta^2)*v1^3 + 6*vx*(-3 + self.theta)*v1*v2 + 6*vx^2*v3))/Pgas + (vx^2*(-2 + self.theta)*self.theta*v1*(PX^3*(-3 + self.theta)*(-2 + self.theta)*(-1 + self.theta)*v1^4 - 12*(FX - Pgas)*Pgas*(2 - 3*self.theta + self.theta^2)*den*v1^2*v2 + 12*Pgas^2*PX*(-1 + self.theta)*(v2^2 + 2*v1*v3) - 24*Pgas^3*v4))/Pgas^3 - (-2 + self.theta)*self.theta*v1*(((-60 + 47*self.theta - 12*self.theta^2 + self.theta^3)*v1^4)/vx + 12*(12 - 7*self.theta + self.theta^2)*v1^2*v2 + 24*vx*(-3 + self.theta)*v1*v3 + 12*vx*((-3 + self.theta)*v2^2 + 2*vx*v4)) + (vx^3*self.theta* (-(PX^4*(-4 + self.theta)*(-3 + self.theta)*(-2 + self.theta)*(-1 + self.theta)*v1^5) + 20*Pgas*PX^3*(-3 + self.theta)*(-2 + self.theta)*(-1 + self.theta)*v1^3*v2 - 60*Pgas^2*PX^2*(-2 + self.theta)*(-1 + self.theta)*v1*(v2^2 + v1*v3) + 120*Pgas^3*PX*(-1 + self.theta)*(v2*v3 + v1*v4) - 120*Pgas^4*v5))/(5.*Pgas^4)))/ (12.*Pgas) + ((-2 + self.theta)*(((360 - 342*self.theta + 119*self.theta^2 - 18*self.theta^3 + self.theta^4)*v1^5)/60. + (vx*(-60 + 47*self.theta - 12*self.theta^2 + self.theta^3)*v1^3*v2)/3. + vx^2*(12 - 7*self.theta + self.theta^2)*v1^2*v3 + vx^2*(-3 + self.theta)*v1*((-4 + self.theta)*v2^2 + 2*vx*v4) + 2*vx^3*((-3 + self.theta)*v2*v3 + vx*v5)))/vx^3) )/(12.*vx^2*(PX*vx - Pgas*G));
                newv = pade([vx, v1, v2, v3, v4 v5 v6], self.Nt, self.Mt, h);
            else
                v1 = (2*Pgas^self.theta*PX^2*(-BX^2 + PX*vx)^3*self.beta*(-1 + G))/ (vx^2*(BX^8*G + BX^6*(PX*vx*(-2 - 5*G) + 2*FX*G) + 2*PX^3*vx^3*(-(FX*G) + PX*vx*(1 + G)) + BX^4*(-(FY^2*G) - 6*FX*PX*vx*G + PX^2*vx^2*(6 + 9*G)) + BX^2*PX*vx*(FY^2*(-2 + G) + PX*vx*(PX*vx*(-6 - 7*G) + 6*FX*G)))*den^self.theta);
                v2 = (PX*(-BX^2 + PX*vx)^3*(-1 + G)*v1*(-((Pgas^self.theta*PX*self.beta*(-2 + self.theta + (PX*vx*(-1 - (BX^2*FY^2)/(BX^2 - PX*vx)^3)*self.theta)/Pgas))/(vx^3*den^self.theta)) + ((BX^8*(1 + G) - 4*BX^6*PX*vx*(1 + G) + PX^4*vx^4*(1 + G) + BX^4*(FY^2 + 6*PX^2*vx^2)*(1 + G) - BX^2*PX*vx*(FY^2*(-2 + G) + 4*PX^2*vx^2*(1 + G)))*v1)/((BX^2 - PX*vx)^4*(-1 + G))))/ (-(BX^8*G) + BX^4*(PX^2*vx^2*(-6 - 9*G) + FY^2*G + 6*FX*PX*vx*G) - 2*PX^3*vx^3*(-(FX*G) + PX*vx*(1 + G)) + BX^6*(-2*FX*G + PX*vx*(2 + 5*G)) + BX^2*PX*vx*(FY^2*(2 - G) + PX*vx*(-6*FX*G + PX*vx*(6 + 7*G))));
                v3 = (PX*(-BX^2 + PX*vx)^3*(-1 + G)*((4*(BX^8*(1 + G) - 4*BX^6*PX*vx*(1 + G) + PX^4*vx^4*(1 + G) + BX^4*(FY^2 + 6*PX^2*vx^2)*(1 + G) - BX^2*PX*vx*(FY^2*(-2 + G) + 4*PX^2*vx^2*(1 + G)))*v1*v2)/ ((BX^2 - PX*vx)^4*(-1 + G)) + (v1*(2*BX^10*(1 + G)*v2 - 10*BX^8*PX*vx*(1 + G)*v2 - 2*PX^5*vx^5*(1 + G)*v2 + 2*BX^6*(FY^2 + 10*PX^2*vx^2)*(1 + G)*v2 + BX^2*PX^2*vx*(10*PX^2*vx^3*(1 + G)*v2 - FY^2*(-2 + G)*(3*v1^2 - 2*vx*v2)) + BX^4*PX*(-20*PX^2*vx^3*(1 + G)*v2 + FY^2*(3*(2 + G)*v1^2 + 2*vx*(1 - 2*G)*v2))))/((BX^2 - PX*vx)^5*(-1 + G)) - (Pgas^self.theta*PX*self.beta*((2*PX*vx*(-1 - (BX^2*FY^2)/(BX^2 - PX*vx)^3)*(-2 + self.theta)*self.theta*v1^2)/Pgas + (-2 + self.theta)*((-3 + self.theta)*v1^2 + 2*vx*v2) + (PX*vx^2*self.theta*(PX*(1 + (BX^2*FY^2)/(BX^2 - PX*vx)^3)^2*(-1 + self.theta)*v1^2 + Pgas*(-2*v2 - (2*BX^4*FY^2*v2)/(BX^2 - PX*vx)^4 + (BX^2*FY^2*PX*(-3*v1^2 + 2*vx*v2))/(BX^2 - PX*vx)^4)))/ Pgas^2))/(vx^4*den^self.theta)))/ (3.*(-(BX^8*G) + BX^4*(PX^2*vx^2*(-6 - 9*G) + FY^2*G + 6*FX*PX*vx*G) - 2*PX^3*vx^3*(-(FX*G) + PX*vx*(1 + G)) + BX^6*(-2*FX*G + PX*vx*(2 + 5*G)) + BX^2*PX*vx*(FY^2*(2 - G) + PX*vx*(-6*FX*G + PX*vx*(6 + 7*G)))));
                v4 = ((-BX^2 + PX*vx)^3*(-1 + G)*((PX*v2*(2*BX^10*(1 + G)*v2 - 10*BX^8*PX*vx*(1 + G)*v2 - 2*PX^5*vx^5*(1 + G)*v2 + 2*BX^6*(FY^2 + 10*PX^2*vx^2)*(1 + G)*v2 + BX^2*PX^2*vx*(10*PX^2*vx^3*(1 + G)*v2 - FY^2*(-2 + G)*(3*v1^2 - 2*vx*v2)) + BX^4*PX*(-20*PX^2*vx^3*(1 + G)*v2 + FY^2*(3*(2 + G)*v1^2 + 2*vx*(1 - 2*G)*v2))))/((BX^2 - PX*vx)^5*(-1 + G)) + (3*PX*(BX^8*(1 + G) - 4*BX^6*PX*vx*(1 + G) + PX^4*vx^4*(1 + G) + BX^4*(FY^2 + 6*PX^2*vx^2)*(1 + G) - BX^2*PX*vx*(FY^2*(-2 + G) + 4*PX^2*vx^2*(1 + G)))*v1*v3)/((BX^2 - PX*vx)^4*(-1 + G)) + (PX*v1*(BX^12*(1 + G)*v3 - 6*BX^10*PX*vx*(1 + G)*v3 + PX^6*vx^6*(1 + G)*v3 + BX^8*(FY^2 + 15*PX^2*vx^2)*(1 + G)*v3 - BX^2*PX^3*vx*(6*PX^2*vx^4*(1 + G)*v3 + FY^2*(-2 + G)*(2*v1^3 - 3*vx*v1*v2 + vx^2*v3)) + BX^4*PX^2*(15*PX^2*vx^4*(1 + G)*v3 + FY^2*(2*(3 + G)*v1^3 - 6*vx*G*v1*v2 + 3*vx^2*(-1 + G)*v3)) + BX^6*PX*(-20*PX^2*vx^3*(1 + G)*v3 + 3*FY^2*((2 + G)*v1*v2 - vx*G*v3))))/((BX^2 - PX*vx)^6*(-1 + G)) - (Pgas^self.theta*PX^2*self.beta*((3*PX*vx*(-1 - (BX^2*FY^2)/(BX^2 - PX*vx)^3)*(-2 + self.theta)*self.theta*v1*((-3 + self.theta)*v1^2 + 2*vx*v2))/Pgas + (3*vx^2*(-2 + self.theta)*self.theta*v1*(PX^2*(1 + (BX^2*FY^2)/(BX^2 - PX*vx)^3)^2*(-1 + self.theta)*v1^2 + 2*Pgas*(-(PX*v2) - (BX^2*FY^2*PX*(3*PX*v1^2 + 2*BX^2*v2 - 2*PX*vx*v2))/(2.*(BX^2 - PX*vx)^4))))/Pgas^2 + (-2 + self.theta)*((12 - 7*self.theta + self.theta^2)*v1^3 + 6*vx*(-3 + self.theta)*v1*v2 + 6*vx^2*v3) + (vx^3*self.theta*(-(PX^3*(1 + (BX^2*FY^2)/(BX^2 - PX*vx)^3)^3*(-2 + self.theta)*(-1 + self.theta)*v1^3) + 6*Pgas*PX*(-1 - (BX^2*FY^2)/(BX^2 - PX*vx)^3)*(-1 + self.theta)*v1* (-(PX*v2) - (BX^2*FY^2*PX*(3*PX*v1^2 + 2*BX^2*v2 - 2*PX*vx*v2))/(2.*(BX^2 - PX*vx)^4)) + 6*Pgas^2*(-(PX*v3) - (BX^2*FY^2*PX* (BX^4*v3 + BX^2*PX*(3*v1*v2 - 2*vx*v3) + PX^2*(2*v1^3 - 3*vx*v1*v2 + vx^2*v3)))/(BX^2 - PX*vx)^5)))/ Pgas^3))/(6.*vx^5*den^self.theta)))/ (2.*(-(BX^8*G) + BX^4*(PX^2*vx^2*(-6 - 9*G) + FY^2*G + 6*FX*PX*vx*G) - 2*PX^3*vx^3*(-(FX*G) + PX*vx*(1 + G)) + BX^6*(-2*FX*G + PX*vx*(2 + 5*G)) + BX^2*PX*vx*(FY^2*(2 - G) + PX*vx*(-6*FX*G + PX*vx*(6 + 7*G)))));
                newv = pade([vx, v1, v2, v3, v4], self.Nt, self.Mt, h);
            end

            vx = newv;

            v1 = self.calculateVprime(vx);

            % If this has the same sign as vx, signal OH CRAP before returning
            if((v1*vx > 0) || (newv < 0)); self.integErrorFlag = 1; end;

        end

        % Evolves the solution using the 5th order Adams-Moulton integrator
        function vnew = takeAB5Step(self, vx, dx)
            vnew = vx + dx*(self.ABHistory*[251 -1274 2616 -2774 1901]')/720;
            
            v1 = self.calculateVprime(vnew);

            self.ABHistory = [self.ABHistory(2:5) v1];

            if ((v1*vx > 0) || (vnew < 0)); self.integErrorFlag = 1; end
        end

        % Calculates the flow state (rho, vx, P) on an interval of width X using
        % an initial step of dx
        function dsingularity = CalculateFlowTable(self, vx, h, Lmax)
            nRestarts = 0;
            self.flowSolution = [0 vx];
            
            dvdx_first = abs(self.calculateVprime(vx));
            
            self.restartAdamsBashforth(vx, h);

            while (self.flowSolution(end,1) < Lmax) && (nRestarts < 7)
                newv = self.takeAB5Step(self.flowSolution(end,2), h);

                % Adaptively monitor stepsize for problems:
                % Rapidly decrease in event of error
                if ( (self.integErrorFlag == 1) || (newv < 0) || (isreal(newv) == 0) )
                    self.flowSolution = self.flowSolution(1:(end-5),:);

                    h=h/16;
                    self.restartAdamsBashforth(self.flowSolution(end,2), h);
                    nRestarts = nRestarts + 1;
                    self.integErrorFlag = 0;
                else
                    self.flowSolution = [self.flowSolution; [self.flowSolution(end,1)+h, newv]];
                    % But conservative increase in event of boringness
                    if (self.flowSolution(end,2) > 1000*abs(self.ABHistory(end))*h)
                        h = 1.5*h;
                        self.restartAdamsBashforth(self.flowSolution(end,2), h);
                    end
                end

                % The above timestep strategy will succeed because we know the characteristic
                % behavior of our flows, either the solution's curvature becomes singular in 
                % finite distance or it asymptotes to zero slope
                % But just in case...
                if mod(numel(self.flowSolution),20000) == 0
                    fprintf('Took %i steps without reaching singularity or terminating; x=%.15f; vx=%.15f\n', numel(self.flowSolution)/2, self.flowSolution(end,1), self.flowSolution(end,2));
                end
%                if abs(self.ABHistory(5)) < .00001*dvdx_first;
%                    fprintf('Ended with v'' < .00001v''0');
%                    break;
%                end
            end

            dsingularity = self.flowSolution(end,1);
        end

    end

end

