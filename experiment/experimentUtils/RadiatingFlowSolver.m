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
        vxTerminal; %velocity at which integration ends

        isHydro;
        ABHistory; % Previous values of vx' used by the AB integrator
        step;
    end

    properties (SetAccess = private, GetAccess = public)
        flowSolution;
        Tcutoff;
    end

    methods (Access = public)
        % Class constructor
        function self = RadiatingFlowSolver(rho, vx, vy, bx, by, P, gamma, beta, theta, Tmin)
% RadiatingFlowSolver class
% RFS(rho, vx, vy, bx, by, P, gamma, beta, theta) sets up to solve a radiating flow's equilibrium 
% with the given initial conditions.
            if(nargin < 10)
               disp('Require:  RadiatingFlowSolver(rho, vx, vy, bx, by, P, gamma, beta, theta, Tcutoff)');
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
            self.Tcutoff = Tmin;
            self.vxTerminal = 0;

            self.px = self.vx0 * self.rho0;
            self.fx = self.px*self.vx0 + self.Pgas0 +(by^2-bx^2)/2;
            self.fy = rho*vx*vy-bx*by;

            self.numericalSetup(1,1);
            self.solver('AB5');
        end

        % Solvers for primitives based on conserved quantities
        function x = rho(self, vx);    x = self.px ./ vx;        return; end
        function x = by(self, vx);     x = (self.bx0 * self.fy)./(-self.bx0^2 + self.px * vx); return; end
        function x = vy(self, vx);     x = (self.fy * vx)./(-self.bx0^2 + self.px * vx); return; end
        function x = Ptherm(self, vx); x = self.fx - self.px * vx + .5*self.bx0^2 *(1 - self.fy^2./(self.bx0^2 - self.px * vx).^2); return; end
        function x = Pmag(self, vx);   x = (self.bx0^2 + self.by(vx)^2)/2; return; end

        function numericalSetup(self, N, M)
            self.Nt = N;
            self.Mt = M;
        end

        function solver(self, id)
            if strcmp(id, 'taylor') == 1; self.step = @self.takeTaylorStep; end
            if strcmp(id, 'AB5') == 1; self.step = @self.takeAB5Step; end
            if strcmp(id, 'AM5') == 1; self.step = @self.takeAM5Step; end
            if strcmp(id, 'AM6') == 1; self.step = @self.takeAM6Step; end
        end

        function help(self)
fprintf('Help for the radiating flow solver:\n\nInitialize with R = RadiatingFlowSolver(rho, vx, vy, bx, by, P, gamma, beta, theta, Tcutoff).\n  R.calculateFlowtable(vx0) solves the differential equation starting at vx0.\n  R.coolingLength(vx) gives the instantaneous cooling length evaluated at vx.\n  R.coolingTime(vx) gives the instantaneous characterisic cooling time evaluated at vx.\n  R.magCutoffV(), if the flow is magnetized, returns the magnetic asymptotic velocity.\n  R.thermalCutoffV(T) returns the velocity when P/rho drops to T.\n  R.temperature(vx) returns the temperature when the flow has velocity vx.\n  R.solutionTable returns an Nx7 matrix with rows [x rho vx vy bx by Pgas].\n');
        end

        function s = solutionTable(self)
            s = zeros(size(self.flowSolution,1),7);
            v = self.flowSolution(:,2);

            s(:,1) = self.flowSolution(:,1);
            s(:,2) = self.rho(v);
            s(:,3) = v;
            s(:,4) = self.vy(v);
            s(:,5) = self.bx0;
            s(:,6) = self.by(v);
            s(:,7) = self.Ptherm(v);
        end

        function lx = coolingLength(self, v)
        % Returns the instantaneous cooling length given vx=v
            if nargin == 1; % vx not given
                v = self.vx0;
            end

            einternal = self.Ptherm(v) / (self.gamma-1.0);
            radrate   = self.beta * self.rho(v)^(2.0-self.theta) * self.Ptherm(v)^self.theta;

            lx = v * einternal / radrate;
        end
        
        function tcool = coolingTime(self, v)
        % Returns the instantaneous cooling time given vx=v
            einternal = self.Ptherm(v) / (self.gamma-1.0);
            radrate   = self.beta * self.rho(v)^(2.0-self.theta) * self.Ptherm(v)^self.theta;
            
            tcool = einternal/radrate;
        end


        function T = temperature(self, v)
        % Returns the temperature at vx=v via the ideal gas relation T = P mu / rho R
        % Assuming mu = R = 1.
            T = self.Ptherm(v) ./ self.rho(v);
        end

        function vb = magCutoffV(self)
        % Returns the velocity to which a magnetized flow will asymptote
        % These are the coefficients such that thermal pressure = 0.
            a = -2*self.px^3;
            b = 5*self.bx0^2*self.px^2 + 2*self.fx*self.px^2;
            c = -4*self.bx0^4*self.px - 4*self.bx0^2*self.fx*self.px;
            d = (self.bx0^6 + 2*self.bx0^4*self.fx-self.bx0^2*self.fy^2);

            vb = sort(roots([a b c d]));
            % Require that the flow actually be going slower...
            vb = vb(vb < self.vx0);
            % Then by the intermediate value theorem...
            vb = max(vb);
        end

        function vt = thermalCutoffV(self, T)
        % Returns the velocity at which temperature == P/rho will drop to T.
            if self.isHydro
                vt(1) = (self.fx + sqrt(self.fx^2 - 4*self.px^2*T))/(2*self.px);
                vt(2) = (self.fx - sqrt(self.fx^2 - 4*self.px^2*T))/(2*self.px);
           else
                a = -2*self.px^3;
                b = 5*self.bx0^2*self.px^2 + 2*self.fx*self.px^2;
                c = -4*self.bx0^4*self.px - 4*self.bx0^2*self.fx*self.px - 2*self.px^3*T;
                d = self.bx0^2*(self.bx0^4 + 2*self.bx0^2*self.fx - self.fy^2 + 4*self.px^2*T);
                e = -2*self.bx0^4*self.px*T;
                vt = solveQuartic(a, b, c, d, e);
            end

            vt = sort(real(vt));
            vt = vt(vt < self.vx0);
            vt = max(vt);

        end

        function setCutoff(self, type, value)
        %depending on type, set vxCutoff to thermalCutoff or magneticCutoff
            if strcmp(type,'none')
                self.vxTerminal = 0;
            end
            if strcmp(type,'magnetic')
                self.vxTerminal = value*self.magCutoffV();
            end
            if strcmp(type,'thermal')
                self.vxTerminal = self.thermalCutoffV(value);
            end

            fprintf('Terminal velocity set by %s to %.10g.\n',type,self.vxTerminal);
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
        function restartLMM(self, vx, h)
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

        end

        % Evolves the solution using the 5th order Adams-Bashforth integrator
        % Coefficients on f'_0 to f'_-4 are [1901 -2274 2616 1274 251]/720
        function vnew = takeAB5Step(self, vx, dx)
            vnew = vx + dx*(self.ABHistory*[251 -1274 2616 -2774 1901]')/720;
            self.ABHistory = [self.ABHistory(2:5) self.calculateVprime(vnew)];
        end

        % Evolves the solution using the 6th order Adams-Moulton integrator
        % Both of these implicit solvers make the prediction using the AB5 method
        % And refine with a few Newton-Raphson iterations
        % Coefficients on f'_1 to f'_-3 are [646 -264 102 -19]/720
        function vnew = takeAM5Step(self, vx, dx)
            vb = vx + dx*(self.ABHistory*[251; -1274; 2616; -2774; 1901])/720; % Prediction
            va = vx;
            % y-independent part of y'=f(x, y)
            G = vx + dx*(self.ABHistory(2:5)*[-19; 106; -264; 646])/720;

            fnc = @(x) x - 251*dx*self.calculateVprime(x)/720 - G;
            
            for N = 1:3
                vc = vb - fnc(vb) *imag(1e-8*1i*vb)/imag(fnc(vb*(1+1i*1e-8)));
                va = vb; vb = vc;
            end

            vnew = vb;
            self.ABHistory = [self.ABHistory(2:5) self.calculateVprime(vnew)];
        end

        % Evolves the solution using the 7th order Adams-Moulton integrator
        % Coefficients on f'_1 to f'_-4 are [1427 -798 482 -173 27]/1440
        function vnew = takeAM6Step(self, vx, dx)
            vb = vx + dx*(self.ABHistory*[251; -1274; 2616; -2774; 1901])/720; % AB5 Prediction
            va = vx;
            G = vx + dx*(self.ABHistory(1:5)*[27; -173; 482; -798; 1427])/1440;

            fnc = @(x) x - 95*dx*self.calculateVprime(x)/288 - G;
            
            for N = 1:3
                vc = vb - fnc(vb) *imag(1e-8*1i*vb)/imag(fnc(vb*(1+1i*1e-8)));
                va = vb; vb = vc;
            end

            vnew = vb;
            self.ABHistory = [self.ABHistory(2:5) self.calculateVprime(vnew)];
        end
        
        % Calculates the flow state (rho, vx, P) on an interval of width X using
        % an initial step of dx
        function dsingularity = calculateFlowTable(self, vx, h, Lmax)
            nRestarts = 0;
            self.flowSolution = [0 vx];
            
            self.restartLMM(vx, h);

            while (self.flowSolution(end,1) < Lmax) && (nRestarts < 15)
                oldv = self.flowSolution(end,2);

                newv = self.step(oldv, h);
                vp   = self.calculateVprime(newv);
                % Error conditions: New v has larger magnitude, new v' would accelerate,
                % or new v has opposite sign
                ohCrap = (abs(newv/oldv) > 1) || (vp * newv > 0) || (newv * oldv < 0) || ~isreal(newv);

                % Adaptively monitor stepsize for problems:
                % Back up and decrease in event of error
                if ohCrap
                    self.flowSolution = self.flowSolution(1:(end-7),:);

                    h=h/8;
                    self.restartLMM(self.flowSolution(end,2), h);
                    nRestarts = nRestarts + 1;
                    newv = self.flowSolution(end,2);
                else
                    self.flowSolution = [self.flowSolution; [self.flowSolution(end,1)+h, newv]];
                    % But conservatively increase if solution becomes flat
                    if (self.flowSolution(end,2) > 1000*abs(self.ABHistory(end))*h)
                        h = 1.5*h;
                        self.restartLMM(self.flowSolution(end,2), h);
                    end
                end

                if newv < self.vxTerminal % Clip off all too-small velocities
                    j = size(self.flowSolution,1);
                    while self.flowSolution(j,2) < self.vxTerminal; j=j-1; end
                    self.flowSolution = self.flowSolution(1:j,:);

		    % Take one more step that should be 'about right' then quit
                    vcur = self.flowSolution(end,2);

                    % Predict how far until reaching terminal velocity
                    hprime = abs((vcur-self.vxTerminal)/self.calculateVprime(vcur));
                    vprime = self.takeTaylorStep(vcur,hprime);
                    self.flowSolution(end+1,:) = [self.flowSolution(end,1)+hprime, self.vxTerminal];
                    break;
                end


                % The above timestep strategy will succeed because we know the characteristic
                % behavior of our flows, either the solution's curvature becomes singular in 
                % finite distance or it asymptotes to zero slope
                % But just in case...
                if mod(numel(self.flowSolution),20000) == 0
                    fprintf('Took %i steps without reaching singularity or terminating; x=%.15f; vx=%.15f\n', numel(self.flowSolution)/2, self.flowSolution(end,1), self.flowSolution(end,2));
                end

            end

            dsingularity = self.flowSolution(end,1);
        end

    end

end

