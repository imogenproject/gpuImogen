classdef FluidWaveGenerator < handle
    % Class annotation template for creating new classes.
    %___________________________________________________________________________________________________
    
    %===================================================================================================
    properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
    end %PUBLIC

    properties (SetAccess = protected, GetAccess = public)
        waveRho;
        waveVel;
        wavePressure;
        waveB;
    end
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
        pRho;
        pV;
        pP;
        pB;
        pGamma;

        % Derived: polytropic K and infinitesmal soundspeed
        polyK;
        cs0;
    end %PROTECTED

    properties (Dependent = true)
        density;
        velocity;
        pressure;
        Bfield;
        gamma;
    end
    
    %===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        % FIXME: Oh man, no checks on anything. Get the duct tape and bailing wire ready
        function r = get.density(self); r = self.pRho; end
        function set.density(self, rho0); self.pRho = rho0; end

        function v = get.velocity(self); v = self.pV; end
        function set.velocity(self, v0); self.pV = v0; end

        function P = get.pressure(self); P = self.pP; end
        function set.pressure(self, P0); self.pP = P0; end

        function B = get.Bfield(self); B = self.pB; end
        function set.Bfield(self, B0); self.pB = B0; end

        function G = get.gamma(self); G = self.pGamma; end
        function set.gamma(self, G); self.pGamma = G; end
    end%GET/SET
    
    %===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]

        function obj = FluidWaveGenerator(iniRho, iniV, iniP, iniB, iniGamma)
            if nargin >= 1; obj.density = iniRho; else; obj.density = 1;        end
            if nargin >= 2; obj.velocity = iniV;  else; obj.velocity = [0 0 0]; end
            if nargin >= 3; obj.pressure = iniP;  else; obj.pressure = 1;       end
            if nargin >= 4; obj.Bfield = iniB;    else; obj.Bfield = [0 0 0];   end
            if nargin >= 5; obj.gamma = iniGamma; else; obj.gamma = 5/3;        end
        end

        function vx = waveVx(self)
            if numel(self.waveVel) == 0; vx = 0; return; end
            vx = squish(self.waveVel(1,:,:,:));
        end 

        function vy = waveVy(self)
            if numel(self.waveVel) == 0; vy = 0; return; end
            vy = squish(self.waveVel(2,:,:,:));
        end 

        function vz = waveVz(self)
            if numel(self.waveVel) == 0; vz = 0; return; end
            vz = squish(self.waveVel(3,:,:,:));
        end

        function vel = waveVelocity(self)
        % vel = waveVelocity() is a utility that computes the velocity primitive
            vel = zeros(size(self.waveVel));
            vel(1,:,:,:) = squish(self.waveVel(1,:,:,:),'onlyleading');
            vel(2,:,:,:) = squish(self.waveVel(2,:,:,:),'onlyleading');
            vel(3,:,:,:) = squish(self.waveVel(3,:,:,:),'onlyleading');
        end
        
        function mom = waveMomentum(self)
        % mom = waveMomentum() is a utility that computes the conserved momentum variable that
        % Imogen runs on from the wave density/velocity functions
            mom = zeros(size(self.waveVel));
            mom(1,:,:,:) = self.waveRho.*squish(self.waveVel(1,:,:,:),'onlyleading');
            mom(2,:,:,:) = self.waveRho.*squish(self.waveVel(2,:,:,:),'onlyleading');
            mom(3,:,:,:) = self.waveRho.*squish(self.waveVel(3,:,:,:),'onlyleading');
        end
        
        function Etot = waveTotalEnergy(self)
            Etot = .5*squish(sum(self.waveVel.^2, 1)) .* self.waveRho + self.wavePressure / (self.pGamma-1);
        end
        
        function Eint = waveInternalEnergy(self)
            Eint = self.wavePressure / (self.pGamma-1);
        end
        
        % I'm prototyping, we'll check for sanity once this turkey takes off
        % Evaluate sonic waves propagating in the direction of k
        % This always uses the +\hat{k} sense; Reverse the wave direction via k -> -k
        function [omega, evector] = sonicInfinitesmal(self, drho, k)
            self.updateDerivedConstants();

            omega = self.cs0 * norm(k);
            evector = [1, self.cs0 / self.pRho, self.cs0^2];
            
            self.waveRho = self.pRho + drho;
            khat = k / norm(k);

            dv = self.cs0*drho/self.pRho;
            self.waveVel = zeros([3 size(dv)]);
            self.waveVel(1,:,:,:) = self.pV(1) + khat(1)*dv;
            self.waveVel(2,:,:,:) = self.pV(2) + khat(2)*dv;
            self.waveVel(3,:,:,:) = self.pV(3) + khat(3)*dv;

            self.wavePressure = self.pP + self.cs0^2 * drho;
        end

        % Integrate the exact sonic characteristic
        function [omega, evector] = sonicExact(self, drho, k)
            self.updateDerivedConstants();
            
            omega = self.cs0 * norm(k);
            evector = [1, self.cs0 / self.pRho, self.cs0^2];

            gm1d2 = (self.pGamma-1)/2;
            khat = k / norm(k);

            self.waveRho = self.pRho + drho;
            % fixme: Prettify and factor infinitesmal c_s out of this
            deltaV = ((self.pRho + drho).^gm1d2 - self.pRho^gm1d2) * sqrt(self.polyK*self.pGamma) / gm1d2;
            self.waveVel = zeros([3 size(deltaV)]);
            self.waveVel(1,:,:,:) = self.pV(1) + khat(1)*deltaV;
            self.waveVel(2,:,:,:) = self.pV(2) + khat(2)*deltaV;
            self.waveVel(3,:,:,:) = self.pV(3) + khat(3)*deltaV;

            self.wavePressure = self.polyK*(self.pRho + drho).^self.pGamma;
        end

        function [omega, evec] = dustyLinear(self, amp, phase, k, forfluid, kDrag, eigensense)
            % Make and solve the 5x5 matrix
            
            M = self.dustyMatrix(self.pRho(1), norm(k), self.pGamma(1), self.pP(1), self.pRho(2), -(self.pRho(1)+self.pRho(2))*kDrag/self.pRho(1));
            
            w0 = sqrt(self.pP(1)/(self.pRho(1)+self.pRho(2)))*norm(k);
            [eigvecs, eigvals] = eig(M);
            % pick the forward-going eigenvalue
            for N = 1:5
                % pick the antisense-propagating wave
                if (eigensense == 1) && (real(eigvals(N,N)) > .01*w0); break; end
                % pick the evanescent wave
                if (eigensense == 0) && (abs(real(eigvals(N,N))) < 1e-12) && (imag(eigvals(N,N)) < 0); break; end
                % pick the prosense-propagating wave
                if (eigensense == -1) && (real(eigvals(N,N)) < -01*w0); break; end
            end
            
            omega = eigvals(N,N);
            evec  = eigvecs(:,N); % [drho dv dP dpsi du]'
            evec = evec / evec(1);

            csq = self.pGamma(1) * self.pP(1) / self.pRho(1);
            
            evec2 = [1, 17350, csq, 2.6525e-3 * 1i, 3.9 * 1i];
            
            if forfluid==1
                qc = 2*pi*kDrag / real(omega);
                SaveManager.logPrint("Dusty wave generator: k_couple/re[f_osc]=%f\n", qc);
                if qc > 10; SaveManager.logPrint('    q > 10: Strong coupling\n'); end
                if (.1 < qc) && (qc < 10); SaveManager.logPrint('    .1 < q < 10: Intermediate coupling\n'); end
                if qc < .1; SaveManager.logPrint('    q < .1: Weak coupling\n'); end
                SaveManager.logPrint("Dusty wave generator: t_damp * re[f_osc]=%f\n", 2*pi*imag(omega) / real(omega));
            end
            %evec = evec2;
            heavdef = 1.0;%*(phase < 12*pi);
            
            %evecb./evec
            if forfluid == 1; q = (evec(1)); else; q = (evec(4)); end
            %if forfluid == 1; q = evec(1); else q = evec(4); end
            self.waveRho = self.pRho(forfluid) + heavdef.*imag(q*amp*exp(1i*phase));
            
            khat = k / norm(k);
            if forfluid == 1; q = (evec(2)); else; q = (evec(5)); end
            %if forfluid == 1; q = evec(2); else q = evec(5); end
            dv = heavdef.*imag(q*amp*exp(1i*phase));
            self.waveVel = zeros([3 size(dv)]);
            self.waveVel(1,:,:,:) = self.pV(1,1) + khat(1)*dv;
            self.waveVel(2,:,:,:) = self.pV(2,1) + khat(2)*dv;
            self.waveVel(3,:,:,:) = self.pV(3,1) + khat(3)*dv;

            % This is meaningful only for the gas, obviously.
            if forfluid == 1
                self.wavePressure = self.pP(1) + heavdef.*imag((evec(3))*amp*exp(1i*phase));
            else
                self.wavePressure = self.pP(2) * ones(size(self.waveRho));
            end
        end
        
        % Entropy wave is linear and has no approximate form
        function [omega, evector] = entropyExact(self, drho, k)
            self.updateDerivedConstants();
            self.waveRho = self.pRho + drho;
            
            self.waveVel = zeros([3 size(drho)]);
            self.waveVel(1,:,:,:) = self.pV(1);
            self.waveVel(2,:,:,:) = self.pV(2);
            self.waveVel(3,:,:,:) = self.pV(3);
            
            self.wavePressure = self.pP * ones(size(self.waveRho));
            
            omega = 0;
            evector = [1 0 0];
        end
        
    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                      P R O T E C T E D    [M]
        function updateDerivedConstants(self)
            % Update polytropic K constant
            self.polyK = self.pP ./ self.pRho.^self.pGamma;

            % Update soundspeed
            self.cs0 = sqrt(self.pGamma * self.pP ./ self.pRho);
        end


    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %                                                 S T A T I C    [M]
        function M = dustyMatrix(rho, wavek, gam, P, psi, kd)
            kd=kd * rho*psi/(rho+psi);
            c = sqrt(gam*P/rho);
            
            M =  [0,   rho*wavek,         0, 0,            0;...
                  0,  -1i*kd/rho, wavek/rho, 0,    1i*kd/rho;...
                  0, wavek*gam*P,         0, 0,            0;...
                  0,           0,         0, 0, psi*wavek;...
                  0,   1i*kd/psi,         0, 0,      -1i*kd/psi];
        end
        
        function f = evalEvecComponent(x, y, z, t, k, omega, a, eigvector, u0, part, fluid)
            phase = k(1)*x+k(2)*y+k(3)*z-omega*t;
            
            switch(part)
                case 1; k = 1;
                case 5; k = 2;
                case 9; k = 3;
                    default; k = 6;
            end
            if fluid.gamma < 1.1; k = k + 3; end
            
            if k > 5
                f = zeros(size(x));
            else
                f = u0(k) + a*imag(eigvector(k)*exp(1i*phase));
            end
        end
    end%PROTECTED
    
end%CLASS
