classdef ShuOsher_ISAnalysis < handle
% Class annotation template for creating new classes.
%___________________________________________________________________________________________________ 

%===================================================================================================
        properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
        
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
	epsilon; % entropy wave amplitude
	M0;      % mach
	gamma;   % copy for reference

	timevals;
	shockX;

    end %PUBLIC

%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
    end %PROTECTED
    
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        end%GET/SET
        
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]

    function obj = ShuOsher_ISAnalysis(run, mass, mom, ener, mag)
	% FIXME: Hardcoded
	obj.epsilon = .2;
	obj.M0 = 3;
	obj.gamma = run.GAMMA;

	forceStop_ParallelIncompatible(); % FIXME: this is not happy :(

    end

    function FrameAnalyzer(obj, run, mass, mom, ener, mag)
	obj.timevals(end+1) = run.time.time;
	obj.shockX(end+1) = obj.SO_findShockPosition(mass);
    end

    function finish(obj, run)

	c0 = sqrt(run.GAMMA);

	inputK = (2*pi)*8; % FIXME: hardcoded
	omega0 = obj.M0 * c0 * inputK;

	% Model shock position: Linear function due to propagation, harmonic modulated by driving term
	shock_x0 = 0.25; % FIXME: hardcoded
	shock_v  = obj.M0 * c0;

	shockHO = obj.shockX - (shock_x0 + shock_v * obj.timevals);

	N = numel(obj.timevals);

	tUniform = (1:(N-1)) * obj.timevals(end) / (N-1);
	xUniform = interp1(obj.timevals, shockHO, tUniform,'linear','extrap');

	xFourier = fft(xUniform);
    
    save('tempResults.mat','tUniform','xUniform','xFourier');

    end
        end%PUBLIC
        
%===================================================================================================        
    methods (Access = protected) %                                      P R O T E C T E D    [M]

	function x = SO_findShockPosition(obj, mass)
	    % exact equilibrium postshock density
	    rhoPost = (obj.M0^2 * (1 + obj.gamma)) / (2*+obj.M0^2*(obj.gamma-1));

	    halfbar = (1+rhoPost)/2;

	    rho = mass.array(:,1,1);

	    % Count backwards until encountering the shock jump
	    % FIXME: Will fail for weak shocks & large epsilon 
	    for N = numel(rho):-1:1
		if rho(N-1) > halfbar; break; end
	    end
	
	    a = rho(N-1); b = rho(N);
	    % Linear extrap the position of the shock based on where the halfway point is
	    x = N-1 + (halfbar - a)/(b-a);
	    x = x / numel(rho);

	end

    end%PROTECTED
                
%===================================================================================================        
        methods (Static = true) %                                                 S T A T I C    [M]
        end%PROTECTED
        
end%CLASS
