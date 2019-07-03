classdef nlpoints_class < handle
    % Class annotation template for creating new classes.
    %___________________________________________________________________________________________________
    
    %===================================================================================================
    properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        leftpoint;
        tdata;
        xdata;
    end %PUBLIC
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
    end %PROTECTED
    
    %===================================================================================================
    methods %                                                                     G E T / S E T  [M]
    end%GET/SET
    
    %===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]

       function self = nlpoints_class(xp, tp, x0)
            self.leftpoint = x0;
            self.tdata = tp;
            self.xdata = xp;
            
            f = gcf();
            f.KeyPressFcn = @self.adjustLeft;
        end

        function adjustLeft(self, hobject, edata, handles)
            if strcmp(edata.Key,'uparrow')
                self.leftpoint = self.leftpoint+1;
                self.replot();
            end
            if strcmp(edata.Key, 'downarrow')
                self.leftpoint = self.leftpoint-1;
                self.replot();
            end
          
            
        end

    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                      P R O T E C T E D    [M]
        
        function replot(self)
            timepts = self.tdata(self.leftpoint:end);
            pospts = self.xdata(self.leftpoint:end)';
            
            [coeffs, resid] = polyfit(timepts, pospts, 1);

            oscil = pospts - (coeffs(1)*timepts + coeffs(2));

            plot(oscil([((end-50):end) (1:50)]));
        end
    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %                                                 S T A T I C    [M]
    end%PROTECTED
    
end%CLASS
