classdef RHD_Plotter < handle
% Class annotation template for creating new classes.
%___________________________________________________________________________________________________ 

%===================================================================================================
        properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
        
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
    end %PUBLIC

%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
    end %PROTECTED
    
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        end%GET/SET
        
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]

    function obj = RHD_Plotter(run, mass, ener, mom, mag)

    end

    function FrameAnalyzer(obj, run, mass, ener, mom, mag)

    figure(1);
    hold off;
    plot(mass.array,'r');
    hold on;
    plot(mom(1).array,'g');
    plot((ener.array - .5*mom(1).array.^2./mass.array)*.6666,'b');
    title(sum(run.time.history)); pause(.01);

    end

    function finish(obj, run)

    end
        end%PUBLIC
        
%===================================================================================================        
        methods (Access = protected) %                                      P R O T E C T E D    [M]
        end%PROTECTED
                
%===================================================================================================        
        methods (Static = true) %                                                 S T A T I C    [M]
        end%PROTECTED
        
end%CLASS
