classdef RealtimePlotter <  LinkedListNode
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

    plotDifference;
    insertPause;

    iterationsPerCall;
    firstCallIteration;
    end %PUBLIC

%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
    end %PROTECTED
    
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        end%GET/SET
        
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]

    function self = RealtimePlotter()
        self = self@LinkedListNode(); % init LL data

    end

    function initialize(self, IC, run, mass, ener, mom, mag)
        forceStop_ParallelIncompatible();
        % Let's be real, any CFD sim large enough to call for MPI isn't realtime anyway...

        self.rho0 = mass.array;
        % Set some defaults if no setup is forthcoming
        self.alpha = ceil(size(self.rho0,1)/2);
        self.beta = ceil(size(self.rho0,2)/2);
        self.gamma = ceil(size(self.rho0,3)/2);

        ticker = ImogenEvent([], self.firstCallIteration, [], @self.FrameAnalyzer);
        ticker.active = 1;
        run.attachEvent(ticker);
    end

    function FrameAnalyzer(self, p, run, mass, ener, mom, mag)
        figure(1);
        plotdat = mass.array;
%plotdat = mom(1).array./mass.array;
        if self.plotDifference; plotdat = plotdat - self.rho0; end
        
        switch(self.plotmode)
            case 1
                plot(plotdat(:,self.beta,self.gamma));
            case 2
                plot(squeeze(plotdat(self.alpha,:,self.gamma)));
            case 3
                plot(squeeze(plotdat(self.alpha,self.beta,:)));
            case 4
                imagesc(plotdat(:,:,self.gamma));
            case 5
                imagesc(squeeze(plotdat(:,self.beta,:)));
            case 6
                imagesc(squeeze(plotdat(self.alpha,:,:)));
        end

       % Rearm
        p.iter = p.iter + self.iterationsPerCall;
        p.active = 1;

        title(sum(run.time.history));
        if self.insertPause; x = input('Enter to continue: '); end
    end

    function finalize(self, run, mass, ener, mom, mag)

    end
        end%PUBLIC
        
%===================================================================================================        
        methods (Access = protected) %                                      P R O T E C T E D    [M]
        end%PROTECTED
                
%===================================================================================================        
        methods (Static = true) %                                                 S T A T I C    [M]
        end%PROTECTED
        
end%CLASS
