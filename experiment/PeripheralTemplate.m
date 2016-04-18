classdef ISAnalyzerTemplate < handle
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
        pErrorMetricHistory;
        pInitialInfo;
    end %PROTECTED
    
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        end%GET/SET
        
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]

    function obj = ISAnalyzerTemplate(run, mass, mom, ener, mag)
        pInitialInfo = mass.array;
        % Or otherwise remember whatever I need from the start.
    end

    function FrameAnalyzer(obj, run, mass, mom, ener, mag)
        % I should examine the current simulation state here, by golly
    end

    function finish(obj, run)
        % disp(objpErrorMetricHistory);
        % Or something
    end
        end%PUBLIC
        
%===================================================================================================        
        methods (Access = protected) %                                      P R O T E C T E D    [M]
        end%PROTECTED
                
%===================================================================================================        
        methods (Static = true) %                                                 S T A T I C    [M]
        end%PROTECTED
        
end%CLASS
