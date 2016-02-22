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
            iniMethod = 2; % defualt hllc
            toMethod = [];
            atstep = [];
        end

        function initialize(self, IC, run, mass, ener, mom, mag)
            run.cfdMethod = self.iniMethod;; % hllc by default

            if ~isempty(self.toMethod)
                if self.toMethod == 1;
                    myEvent = ImogenEvent([], 20, [], @self.flipToHLL);
                end
                if self.toMethod == 2;
                    myEvent = ImogenEvent([], 20, [], @self.flipToHLL);
                end
                
                myEvent.active = 1;
                run.attachEvent(myEvent);
            end

        end

        function finalize(self, run, mass, ener, mom, mag)

        end

    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                      P R O T E C T E D    [M]
    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %                                                 S T A T I C    [M]

        function flipToHLL(p, run, mass, ener, mom, mag)
            run.save.logPrint('CFD method changed to HLL at iteration %i\n', run.time.iteration);
            run.cfdMethod = 1; % toggle to HLL
            p.delete();
        end

        function flipToHLLC(p, run, mass, ener, mom, mag);
            run.save.logPrint('CFD method changed to HLLC at iteration %i\n', run.time.iteration);
            run.cfdMethod = 2; % toggle to HLLC
            p.delete();
        end

    end%PROTECTED
    
end%CLASS
