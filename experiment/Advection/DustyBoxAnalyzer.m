classdef DustyBoxAnalyzer < LinkedListNode;
    % Class annotation template for creating new classes.
    %___________________________________________________________________________________________________
    
    %===================================================================================================
    properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        % The thermodynamic properties & constants that define the solution
        sigmaGas,muGas,sigmaDust,muDust,rhoG,rhoD,vGas,vDust,gammaGas,Pgas;

        % how far to go between analyses in timesteps
        stepsPerPoint;

        analysis; 
    end %PUBLIC
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
        solver;

        p_vStick;
        p_dv;
        p_dvHat;
    end %PROTECTED
    
    %===================================================================================================
    methods %                                                                     G E T / S E T  [M]
    end%GET/SET
    
    %===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
        function self = DustyBoxAnalyzer()
            self = self@LinkedListNode();

            self.stepsPerPoint = -25;
        end

        function initialize(self, IC, run, fluids, mag)
            % assumes we are handing a true DustyBox and the structures are spatially uniform
            % FIXME this is a pretty dumb way to pick the point to analyze...
            self.rhoG = fluids(1).mass.array(6,1,1);
            self.rhoD = fluids(2).mass.array(6,1,1);

            % eat the thermodynamic props from the fluid managers
            self.gammaGas  = fluids(1).gamma;
            self.sigmaGas  = fluids(1).particleSigma;
            self.sigmaDust = fluids(2).particleSigma;
            self.muGas     = fluids(1).particleMu;
            self.muDust    = fluids(2).particleMu;

            vGas  = [fluids(1).mom(1).array(6,1,1) fluids(1).mom(2).array(6,1,1) fluids(1).mom(3).array(6,1,1)] / self.rhoG;
            vDust = [fluids(2).mom(1).array(6,1,1) fluids(2).mom(2).array(6,1,1) fluids(2).mom(3).array(6,1,1)] / self.rhoD;

            % initial pecuilar velocity magnitude & direction vectors
            dv = norm(vGas - vDust);
            dvHat = (vGas - vDust) / dv;
            
            self.analysis(1,:) = [0, dv, dv, 0]; % Initial state entry

            % velocity at t=infty when they stick completely
            vStick = (vGas * self.rhoG + vDust * self.rhoD) / (self.rhoG + self.rhoD);

            % simple conservation arguments give us that
            % vGas = vStick + dvHat * dv * d/(d+g)
            % vDust= vStick - dvHat * dv * g/(g+d)

            press = fluids(1).calcPressureOnCPU();
            Pgas  = press(6,1,1);
            
            % we omit the general arbitrary(v1, v2) case having already solved the sticking velocity above
            self.solver = DustyBoxSolver(self.sigmaGas, self.muGas, self.sigmaDust, self.muDust, self.rhoG, self.rhoD, dv, 0, self.gammaGas, Pgas);
            % save useful info
            self.p_vStick = vStick;
            self.p_dv = dv;
            self.p_dvHat = dvHat;
            self.stepsPerPoint = 1; 

            if(self.stepsPerPoint < 0)
                run.save.logPrint(sprintf('WARNING: DustyBoxAnalyzer stepsPerPoint never set; Defaulting to %i\n', int32(abs(self.stepsPerPoint))));
                self.stepsPerPoint = abs(self.stepsPerPoint);
            end

            myEvent = ImogenEvent([], self.stepsPerPoint, [], @self.analyzeDrag);
            myEvent.armed = 1;
            run.attachEvent(myEvent);

        end

        function analyzeDrag(self, evt, run, fluids, mag)
            
            tFinal = sum(run.time.history);
            
            if 1
                opts = odeset('Reltol',1e-13,'AbsTol',1e-14);
                [tout, yout] = ode113(self.solver.f, [0 tFinal], self.solver.solution(1,2),opts);
                dv_exact = yout(end);
            else % This uses my crappy homebrew ODE solver... why the hell was I let to wander down THAT road?
                self.solver.setInitialCondition(0, self.p_dv);
                
                % Largest safe timestep, or largest that takes 4 steps: whichever is less
                tau = .005/abs(self.solver.computeJacobian);
                N = ceil(tFinal / tau);
                if N < 6; N = 6; end
                self.solver.setStep(tFinal / N);
                %fprintf('%12e\n',self.solver.stepsize)
                
                self.solver.integrate(N, 1);%- 3*self.solver.stepsize);% - .001 * self.solver.stepsize);
                dv_exact = self.solver.solution(end,2);
            end
            
            nvGas  = [fluids(1).mom(1).array(6,1,1) fluids(1).mom(2).array(6,1,1) fluids(1).mom(3).array(6,1,1)] / self.rhoG;
            nvDust = [fluids(2).mom(1).array(6,1,1) fluids(2).mom(2).array(6,1,1) fluids(2).mom(3).array(6,1,1)] / self.rhoD;
            
            dv_numeric = norm(nvGas - nvDust);

            self.analysis(end+1,:) = [tFinal, dv_exact, dv_numeric, dv_exact - dv_numeric];
            
            % rearm to fire again
            evt.iter = evt.iter + self.stepsPerPoint;
            evt.armed = 1; 
        end

        function finalize(self, run, fluids, mag)
            result = struct('time',self.analysis(:,1),'dvExact', self.analysis(:,2), 'dvImogen', self.analysis(:,3), 'error', self.analysis(:,4));
            save([run.paths.save '/drag_analysis.mat'], 'result');
        end

    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                      P R O T E C T E D    [M]
    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %                                                 S T A T I C    [M]

    end%PROTECTED
    
end%CLASS
