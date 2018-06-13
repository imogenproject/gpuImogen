classdef MassConservationAnalyzer < LinkedListNode
    % Class annotation template for creating new classes.
    %___________________________________________________________________________________________________
    
    %===================================================================================================
    properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        analysis; 
        stepsPerCheck;
        plotResult;
    end %PUBLIC
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
        m0;
    end %PROTECTED
    
    %===================================================================================================
    methods %                                                                     G E T / S E T  [M]
    end%GET/SET
    
    %===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
        function self = MassConservationAnalyzer()
            self = self@LinkedListNode();

            self.stepsPerCheck = 1;
            self.plotResult = 0;
        end

        function initialize(self, IC, run, fluids, mag)
            % Called initialize(IC, run, fluids, mag) by ImogenManager at start time for
            % for every peripheral attached to the run.

            myEvent = ImogenEvent([], self.stepsPerCheck, [], @self.computeMass);
            myEvent.armed = 1;
            run.attachEvent(myEvent);

        end

        function computeMass(self, evt, run, fluids, mag)
            % set up by initializer to be called every self.stepsPerCheck iterations
            % computes total mass on the grid assuming all-mirror BCs,
            r = fluids(1).mass.gridSize;

            F = fluids(1).mass.array;

            a = sum(sum(F(5:(end-4),5:(end-4))));
            b = sum(sum(F([4 (end-3)],5:(end-4))));
            c = sum(sum(F(5:(end-4),[4 (end-3)])));
            d = F(4,4) + F(end-3,4) + F(end-3,end-3) + F(4,end-3);

            m = a + .5*(b+c) + .25*d;

            self.analysis(end+1,:) = [run.time.iteration, m];
            
            % rearm event to fire again
            evt.iter = evt.iter + self.stepsPerCheck;
            evt.armed = 1; 
        end

        function finalize(self, run, fluids, mag)
            % called by ImogenManager after exit from fluid iteration loop
            result = struct('time',self.analysis(:,1),'mass', self.analysis(:,2)); %#ok<NASGU>
            save([run.paths.save '/mass_conservation_analysis.mat'], 'result');

            if self.plotResult
                figure();
                t = self.analysis(:,1);
                m = self.analysis(:,2);
                m = (m/m(1)) - 1;
                plot(t, m);
                xlabel('Iteration');
                ylabel('(M(t) - M(t=0)) / M(t=0)');
                title('Mass conservation plot');
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
