classdef DustyBoxAnalyzer < LinkedListNode
    % Class annotation template for creating new classes.
    %___________________________________________________________________________________________________
    
    %===================================================================================================
    properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        % The thermodynamic properties & constants that define the solution
        thermoGas, thermoDust; % Copied in from the fluids(x).thermoDetails structure
        
        % Gas and dust mass density
        rhoG, rhoD;

        % how far to go between analyses in timesteps
        stepsPerPoint;
        
        % The results of the postmortem as they happen
        analysis; 
    end %PUBLIC
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
        p_vStick;
        p_dv;
        p_dvHat;

        P0;
    end %PROTECTED
    
    %===================================================================================================
    methods %                                                                     G E T / S E T  [M]
    end%GET/SET
    
    %===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
        function self = DustyBoxAnalyzer()
            self = self@LinkedListNode();
            self.whatami = 'DustyBoxAnalyzer';
            self.stepsPerPoint = -25;
        end

        function initialize(self, IC, run, fluids, mag) %#ok<INUSL,INUSD>
            % assumes we are handing a true DustyBox and the structures are spatially uniform
            % FIXME this is a pretty dumb way to pick the point to analyze...
            self.rhoG = fluids(1).mass.array(6,1,1);
            self.rhoD = fluids(2).mass.array(6,1,1);

            % eat the thermodynamic props from the fluid managers
            self.thermoGas = fluids(1).thermoDetails;
            self.thermoDust= fluids(2).thermoDetails;

            gasVel  = [fluids(1).mom(1).array(6,1,1) fluids(1).mom(2).array(6,1,1) fluids(1).mom(3).array(6,1,1)] / self.rhoG;
            dustVel = [fluids(2).mom(1).array(6,1,1) fluids(2).mom(2).array(6,1,1) fluids(2).mom(3).array(6,1,1)] / self.rhoD;

            % initial pecuilar velocity magnitude & direction vectors
            dv = norm(gasVel - dustVel);
            dvHat = (gasVel - dustVel) / dv;
            
            self.analysis(1,:) = [0, dv, dv, 0]; % Initial state entry

            % velocity at t=infty when they stick completely
            vStick = (gasVel * self.rhoG + dustVel * self.rhoD) / (self.rhoG + self.rhoD);

            % simple conservation arguments give us that
            % vGas = vStick + dvHat * dv * d/(d+g)
            % vDust= vStick - dvHat * dv * g/(g+d)

            press = fluids(1).calcPressureOnCPU();
            %gasP  = press(6,1,1);

            self.p_vStick = vStick;
            self.p_dv = dv;
            self.p_dvHat = dvHat;
            self.stepsPerPoint = 1; 

            self.P0 = press(6,1,1);

            if(self.stepsPerPoint < 0)
                run.save.logPrint(sprintf('WARNING: DustyBoxAnalyzer stepsPerPoint never set; Defaulting to %i\n', int32(abs(self.stepsPerPoint))));
                self.stepsPerPoint = abs(self.stepsPerPoint);
            end

            myEvent = ImogenEvent([], self.stepsPerPoint, [], @self.analyzeDrag);
            myEvent.armed = 1;
            run.attachEvent(myEvent);
        end

        function a = computeAcceleration(self, t, dv)
            % computes d(delta-v)/dt
            press = self.PofV(dv);
            T = (self.thermoGas.mass * press) / (self.rhoG * self.thermoGas.kBolt);
            
            visc = self.thermoGas.dynViscosity * (T/298.15)^self.thermoGas.viscTindex;
            Rey = self.rhoG * dv * sqrt(self.thermoDust.sigma / pi) / visc;
            
            mfp = self.thermoGas.mass * (T/298.15)^self.thermoGas.sigmaTindex / (self.rhoG * self.thermoGas.sigma * sqrt(2));
            Kn = 2*mfp / sqrt(self.thermoDust.sigma / pi);

            Fone = self.computeCdrag(Rey, Kn);
            a = -Fone * .5 * dv^2 * (self.thermoDust.sigma/4) * (self.rhoG + self.rhoD) / self.thermoDust.mass;
        end

        function analyzeDrag(self, evt, run, fluids, mag) %#ok<INUSD>
            
            tFinal = run.time.time;
            
            % Apply out-of-the-box ODE solver with most stringent error tolerances...
            opts = odeset('Reltol',1e-13,'AbsTol',1e-14);
            
            f = @(t, x) self.computeAcceleration(t, x);
            
            [tout, yout] = ode113(f, [0 tFinal], self.p_dv, opts);
            dv_exact = yout(end);
            
            nvGas  = [fluids(1).mom(1).array(6,1,1) fluids(1).mom(2).array(6,1,1) fluids(1).mom(3).array(6,1,1)] / self.rhoG;
            nvDust = [fluids(2).mom(1).array(6,1,1) fluids(2).mom(2).array(6,1,1) fluids(2).mom(3).array(6,1,1)] / self.rhoD;
            
            dv_numeric = norm(nvGas - nvDust);

            self.analysis(end+1,:) = [tFinal, dv_exact, dv_numeric, dv_numeric/dv_exact - 1];
            
            % rearm to fire again
            evt.iter = evt.iter + self.stepsPerPoint;
            evt.armed = 1; 
        end

        function finalize(self, run, fluids, mag) %#ok<INUSD>
            result = struct('time',self.analysis(:,1),'dvExact', self.analysis(:,2), 'dvImogen', self.analysis(:,3), 'error', self.analysis(:,4)); %#ok<NASGU>
            save([run.paths.save '/drag_analysis.mat'], 'result');
        end

        function P = PofV(self, dv)
            % reduced density
            mu = self.rhoG * self.rhoD / (self.rhoG + self.rhoD);
            
            % hydrostatic part of dissipated relative KE added
            P = self.P0 + (self.thermoGas.gamma-1)*.5*mu*(self.p_dv^2 - dv^2);
        end

    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                      P R O T E C T E D    [M]
    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %                                                 S T A T I C    [M]

        function c = computeCdrag(Rey, Kn)
            %c = (24/Rey + 4*Rey^(-1/3) + .44*Rey/(12000+Rey)) / (1 + 1*Kn*(1.142+.558*exp(-.999/Kn)));
            c = (24 / Rey + 3.6*Rey^-.319 + .4072*Rey/(8710+Rey)) / (1 + Kn*(1.142 + 1*0.558*exp(-0.999/Kn)));
        end
        
    end%PROTECTED
    
end%CLASS

