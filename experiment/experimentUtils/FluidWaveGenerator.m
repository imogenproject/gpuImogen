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

        function mom = waveVelocity(self)
        % mom = waveMomentum() is a utility that computes the conserved momentum variable that
        % Imogen runs on from the wave density/velocity functions
            mom = zeros(size(self.waveVel));
            mom(1,:,:,:) = squish(self.waveVel(1,:,:,:),'onlyleading');
            mom(2,:,:,:) = squish(self.waveVel(2,:,:,:),'onlyleading');
            mom(3,:,:,:) = squish(self.waveVel(3,:,:,:),'onlyleading');
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
        function sonicInfinitesmal(self, drho, k)
            self.updateDerivedConstants();

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
        function sonicExact(self, drho, k)
            self.updateDerivedConstants();
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

        % Entropy wave is linear and has no approximate form
        function entropyExact(self, drho, k)
            self.updateDerivedConstants();
            self.waveRho = self.pRho + drho;
            
            self.waveVel = zeros([3 size(drho)]);
            self.waveVel(1,:,:,:) = self.pV(1);
            self.waveVel(2,:,:,:) = self.pV(2);
            self.waveVel(3,:,:,:) = self.pV(3);
            
            self.wavePressure = self.pP * ones(size(self.waveRho));
        end

    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                      P R O T E C T E D    [M]
        function updateDerivedConstants(self)
            % Update polytropic K constant
            self.polyK = self.pP / self.pRho^self.pGamma;

            % Update soundspeed
            self.cs0 = sqrt(self.pGamma*self.pP /self.pRho);
        end


    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %                                                 S T A T I C    [M]
    end%PROTECTED
    
end%CLASS
