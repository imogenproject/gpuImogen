classdef ImogenEvent < LinkedListNode
    % An event which is polled once per Imogen timestep.
    % If the conditions for the event to be triggered are met, the callbackHandle function is called.
    % the .armed property is set to zero if triggered: 
    %___________________________________________________________________________________________________
    
    %===================================================================================================
    properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        armed;         % if true, checks possible event triggers below
        time;           % if armed and sum(run.time.history) >= time, triggers the callback
        iter;           % if armed and run.time.iteration >= iter, triggers the callback
        testHandle;     % if armed and not [], triggers callback if testHandle(event, run, fluids, mag) == 1 
        callbackHandle; % If event is armed and triggered, callbackHandle(event, run, fluids, mag)
    end %PUBLIC
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
    end %PROTECTED
    
    %===================================================================================================
    methods %                                                                     G E T / S E T  [M]
    end%GET/SET
    
    %===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]

        function self = ImogenEvent(time, iter, handle, callback)
                self = self@LinkedListNode();

                if ~isempty(time); self.time = time; else self.time = NaN; end
                if ~isempty(iter); self.iter = iter; else self.iter = NaN; end
                if ~isempty(handle); self.testHandle = handle; end
                if ~isempty(callback); self.callbackHandle = callback; end

                self.armed = 0;
        end

    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                      P R O T E C T E D    [M]
    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %                                                 S T A T I C    [M]
    end%PROTECTED
    
end%CLASS
