classdef pointGetter < handle
    % Class annotation template for creating new classes.
    %___________________________________________________________________________________________________
    
    %===================================================================================================
    properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        clickedPoint;
        ran;
    end %PUBLIC
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
    end %PROTECTED
    
    %===================================================================================================
    methods %                                                                     G E T / S E T  [M]
    end%GET/SET
    
    %===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]

       function self = pointGetter(thetype)
           
            ax = gca();
            im = [];
            for j = 1:numel(ax.Children)
                if strcmp(ax.Children(j).Type,thetype)
                    im = ax.Children(j);
                end
            end
            self.clickedPoint = [];
            im.ButtonDownFcn = @self.clicky;
            self.ran = rand(); 
        end

        function clicky(self, hobject, edata)
            self.clickedPoint = edata.IntersectionPoint(1:2);
            
        end
        
        function pt = waitForClick(self)
            while isempty(self.clickedPoint)
                pause(.25);
            end
            pt = self.clickedPoint;
            self.clickedPoint = [];
        end

    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                      P R O T E C T E D    [M]
        
    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %                                                 S T A T I C    [M]
    end%PROTECTED
    
end%CLASS

