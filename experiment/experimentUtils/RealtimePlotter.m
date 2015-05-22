classdef RealtimePlotter < handle
% Class annotation template for creating new classes.
%___________________________________________________________________________________________________ 

%===================================================================================================
        properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
        
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
    rho0;

    alpha;
    beta;
    gamma;

    plotmode;
    % 1: plot(rho(:,beta,gamma));
    % 2: plot(rho(alpha,:,gamma));
    % 3: plot(rho(alpha,beta,:));
    % 4: imagesc(rho(:,:,gamma));
    % 5: imagesc(rho(:,beta,:));
    % 6: imagesc(rho(alpha,:,:));

    insertPause;
    end %PUBLIC

%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
    end %PROTECTED
    
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        end%GET/SET
        
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]

    function obj = RealtimePlotter(run, mass, mom, ener, mag)

	forceStop_ParallelIncompatible();
	% Let's be real, any CFD sim large enough to call for MPI isn't realtime anyway...

        obj.rho0 = mass.array;
        % Set some defaults if no setup is forthcoming
        obj.alpha = ceil(size(obj.rho0,1)/2);
	obj.beta = ceil(size(obj.rho0,2)/2);
	obj.gamma = ceil(size(obj.rho0,3)/2);
        obj.plotmode = 1;

	obj.insertPause = 0;
    end

    function setup(obj, instructions)
	if isfield(instructions, 'alpha'); obj.alpha = instructions.alpha; end
	if isfield(instructions, 'beta'); obj.beta = insructions.beta; end;
	if isfield(instructions, 'gamma'); obj.gamma = instructions.gamma; end;
	if isfield(instructions, 'plotmode'); obj.plotmode = instructions.plotmode; end
	if isfield(instructions, 'pause'); obj.insertPause = true; end
    end

    function FrameAnalyzer(obj, run, mass, mom, ener, mag)
        figure(1);
        switch(obj.plotmode)
	    case 1
		plot(mass.array(:,obj.beta,obj.gamma));
	    case 2
		plot(squeeze(mass.array(obj.alpha,:,obj.gamma)));
	    case 3
		plot(squeeze(mass.array(obj.alpha,obj.beta,:)));
	    case 4
		imagesc(mass.array(:,:,obj.gamma));
	    case 5
		imagesc(squeeze(mass.array(:,obj.beta,:)));
	    case 6
		imagesc(squeeze(mass.array(obj.alpha,:,:)));
	end

    title(sum(run.time.history));

    if obj.insertPause; x = input('Enter to continue: '); end

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
