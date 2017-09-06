classdef BoosterPeripheral < LinkedListNode;
    % Class annotation template for creating new classes.
    %___________________________________________________________________________________________________
    
    %===================================================================================================
    properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
        TYPE_ONCE = 1;
        TYPE_ALTERNATE = 2;

    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        type
        myProperty;
    end %PUBLIC
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
    end %PROTECTED
    
    %===================================================================================================
    methods %                                                                     G E T / S E T  [M]
    end%GET/SET
    
    %===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
        function self = BoosterPeripheral()
            self = self@LinkedListNode();

            myProperty = 0;
        end

        function initialize(self, IC, run, fluid, mag)

            % An ImogenEvent is a linked list node, a trigger condition & a callback duct taped together
            myEvent = ImogenEvent([], 20, [], @self.callbackMethod);
            myEvent.armed = 1;
            % Initialize is called before the CFD loop begins
            run.attachEvent(myEvent);
        end

        function finalize(self, run, fluids, mag)

        end

    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                      P R O T E C T E D    [M]
    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %                                                 S T A T I C    [M]
        function callbackMethod(p, run, fluids, mag)
            run.save.logPrint('My template method was called!\n');
        end

    end%PROTECTED
    
end%CLASS
