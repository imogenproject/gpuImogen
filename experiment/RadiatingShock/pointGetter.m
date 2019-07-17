classdef pointGetter < handle
    % Class annotation template for creating new classes.
    %___________________________________________________________________________________________________
    
    %===================================================================================================
    properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        clickedPoint;
    end %PUBLIC
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
    end %PROTECTED
    
    %===================================================================================================
    methods %                                                                     G E T / S E T  [M]
    end%GET/SET
    
    %===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]

       function self = pointGetter()
           
            ax = gca();
            im = ax.Children(1);
            im.ButtonDownFcn = @self.clicky;
            
            self.clickedPoint = [];
        end

        function clicky(self, hobject, edata)
            
            self.clickedPoint = edata.IntersectionPoint(1:2);
        end
        
        function pt = waitForClick(self)
            while isempty(self.clickedPoint); pause(.25); end
            pt = self.clickedPoint;
        end

    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                      P R O T E C T E D    [M]
        
    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %                                                 S T A T I C    [M]
    end%PROTECTED
    
end%CLASS

