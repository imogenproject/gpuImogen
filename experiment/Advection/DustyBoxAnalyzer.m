classdef FlipMethod < LinkedListNode;
    % Class annotation template for creating new classes.
    %___________________________________________________________________________________________________
    
    %===================================================================================================
    properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        iniMethod;
        toMethod;
        atstep;

    end %PUBLIC
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
    end %PROTECTED
    
    %===================================================================================================
    methods %                                                                     G E T / S E T  [M]
    end%GET/SET
    
    %===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
        function self = FlipMethod()
            self = self@LinkedListNode();
            self.iniMethod = 2; % default hllc
            self.toMethod = [];
            self.atstep = [];
        end

        function initialize(self, IC, run, fluids, mag)
            run.cfdMethod = self.iniMethod;; % hllc by default

            if ~isempty(self.toMethod)
                if self.toMethod == 1;
                    myEvent = ImogenEvent([], self.atstep, [], @self.flipToHLL);
                end
                if self.toMethod == 2;
                    myEvent = ImogenEvent([], self.atstep, [], @self.flipToHLLC);
                end
                if self.toMethod == 3;
                    myEvent = ImogenEvent([], self.atstep, [], @self.flipToXJ); 
                end
                
                myEvent.active = 1;
                run.attachEvent(myEvent);
            end

        end

        function finalize(self, run, fluids, mag)
            run.save.logPrint('Method flipper: Finalize called.\n');
        end

    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                      P R O T E C T E D    [M]
    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %                                                 S T A T I C    [M]

        function flipToHLL(p, run, fluids, mag)
            run.save.logPrint('CFD method changed to HLL at iteration %i\n', run.time.iteration);
            run.cfdMethod = 1; % toggle to HLL
            p.delete();
        end

        function flipToHLLC(p, run, fluids, mag);
            run.save.logPrint('CFD method changed to HLLC at iteration %i\n', run.time.iteration);
            run.cfdMethod = 2; % toggle to HLLC
            p.delete();
        end
        
        function flipToXJ(p, run, fluids, mag);
            run.save.logPrint('CFD method changed to XinJin at iteration %i\n', run.time.iteration);
            run.cfdMethod = 3; % toggle to HLLC
            p.delete();
        end

    end%PROTECTED
    
end%CLASS
