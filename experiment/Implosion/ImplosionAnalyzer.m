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

	function obj = FrameAnalyzer(mass,ener,momX,momY,momZ,timeStruct)
	
	mass = mass.array(end:-1:1,:);

	calculatedAsymmetry = norm((mass - mass'),'fro'); 
	asymmetryNorm(end+1) = calculatedAsymmetry;



	time(end+1) = sum(timeStruct.history);
	end

	function result = ImplosionAnalyzer()

	end
        end%PUBLIC
       
%===================================================================================================        
        methods (Access = protected) %                                      P R O T E C T E D    [M]
        end%PROTECTED
                
%==================================================================================================        
        methods (Static = true) %                                                 S T A T I C    [M]

	function singleObj = getInstance()
            persistent instance;
	    if isempty(instance) || ~isvalid(instance)
	        instance = ImplosionAnalyzer();
	    end
	    singleObj = instance;
    	end

    end%PROTECTED
        
end%CLASS
