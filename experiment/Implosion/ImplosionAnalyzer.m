classdef ImplosionAnalyzer < handle
% Analyzer to compress meaningful data into small chunks that don't overload HDD
%___________________________________________________________________________________________________ 

%===================================================================================================
        properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
        
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]

	frames;
	asymmetryNorm;
	time;

    end %PUBLIC

%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
    end %PROTECTED
    
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        end%GET/SET
        
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]

	function FrameAnalyzer(obj, mass, ener, mom, mag, run)
            m = mass.array;

            calculatedAsymmetry = norm((m - m'),'fro');
            obj.asymmetryNorm(end+1) = calculatedAsymmetry;
            obj.time(end+1) = run.time.time;
	end

	function result = ImplosionAnalyzer()

    end
    
    function finish(obj, run)
        implode.time = obj.time;
        % mult by h^2 scales sqrt(sum(Mij^2)) to be resolution-invariant
        implode.asymmetry = obj.asymmetryNorm * run.geometry.d3h(1)^2; %#ok<STRNU>
        
        save([run.paths.save '/asymmetryTracking.mat'], 'implode');
    end

        end%PUBLIC
       
%===================================================================================================        
        methods (Access = protected) %                                      P R O T E C T E D    [M]
        end%PROTECTED
                
%==================================================================================================        
        methods (Static = true) %                                                 S T A T I C    [M]

    end%PROTECTED
        
end%CLASS
