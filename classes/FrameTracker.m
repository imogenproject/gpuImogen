classdef FrameTracker < handle
    % Class annotation template for creating new classes.
    %___________________________________________________________________________________________________
    
    %===================================================================================================
    properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        omega;
	rotateCenter;

	centerVelocity;
    end %PUBLIC
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
	% Record the frame's velocity & rotation rates at every step
	velocityHistory;
	omegaHistory;

	% Keep a running integral of the accumulated offset & rotation
	pPositionHistory;
	pAngleHistory;

	% But only update it when its needed: remember last time
	pLastIntegrated;

    end %PROTECTED
    
    %===================================================================================================
    methods %                                                                     G E T / S E T  [M]
%	function x = get.positionHistory(self)
%	end
%
%	function theta = get.angleHistory(self)
%	end
    end%GET/SET
    
    %===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
        function self = FrameTracker()
            % Called by ImogenManager constructor
            self.iniParameters([0 0 0], 0, [0 0]);
        end
        
        function iniParameters(self, iniVelocity, iniOmega, iniCenter)
            % Called by initialize.m:355
        end

	function initialize(self, run, frameParameters, mass, ener, mom)
        % called by ImogenManager.initialize()
        % The alter routines assume these values are what we are at, so we pretend they are zero and it resets them for us.
	    self.omega = frameParameters.omega;
	    self.rotateCenter = frameParameters.rotateCenter;
	    self.centerVelocity = frameParameters.velocity;

            if self.omega ~= 0
                j = self.omega;
                self.omega = 0;
                source_alterFrameRotation(self, run, mass, ener, mom, j);
            end

            if self.centerVelocity ~= 0
                j = self.centerVelocity;
                self.centerVelocity = 0;
                source_alterGalileanBoost(self, mass, ener, mom, j);

            end

            self.pPositionHistory = self.rotateCenter; % ???
            self.pAngleHistory = 0; % arbitrary

	end


    function changeRotationCenter(self, mass, ener, mom, newCenter)
        
    end
    
    function integrateMotion(self, dt)
        self.pAngleHistory(end+1) = self.pAngleHistory(end) + dt*self.omega;
    end
    
    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                      P R O T E C T E D    [M]
    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %                                                 S T A T I C    [M]
    end%PROTECTED
    
end%CLASS
