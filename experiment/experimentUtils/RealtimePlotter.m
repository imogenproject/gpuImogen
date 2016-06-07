classdef RealtimePlotter <  LinkedListNode
% A semi-interactive peripheral for displaying the state of a simulation as
% it runs, in realtime. Usable only by node-serial simulations, and throws
% an error if run in a parallel sim. 
%___________________________________________________________________________________________________
    
%===================================================================================================
    properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        q0;
        cut;

        plotmode;
        % 1: plot(rho(:,beta,gamma));
        % 2: plot(rho(alpha,:,gamma));
        % 3: plot(rho(alpha,beta,:));
        % 4: imagesc(rho(:,:,gamma));
        % 5: imagesc(rho(:,beta,:));
        % 6: imagesc(rho(alpha,:,:));
        % 7: for debug, call 'plotem'
        
        plotDifference;
        insertPause;
        
        iterationsPerCall;
        firstCallIteration;

        generateTeletextPlots;

	forceRedraw;
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
            

            self.q0                 = [];
            self.plotmode           = 4; % Default to mid-Z image
            self.insertPause        = 0;
        
            self.iterationsPerCall  = 100;
            self.firstCallIteration = 1;

            self.cut = [1 1 1];

            self.generateTeletextPlots = 0; % FIXME: set this depending on availability of graphics
	    self.forceRedraw           = 0;
        end
        
        function initialize(self, IC, run, fluids, mag)
            forceStop_ParallelIncompatible();
            % Let's be real, any CFD sim large enough to call for MPI isn't realtime anyway...
            
            numFluids = numel(fluids);
            
            for i = 1:numFluids;
                self.q0{i} = fluids(i).mass.array;
            end
            

            self.cut=[0 0 0];
            for i=1:3; self.cut(i) = ceil(size(self.q0{1},i)/2); end
            
            ticker = ImogenEvent([], self.firstCallIteration, [], @self.FrameAnalyzer);
            ticker.active = 1;
            run.attachEvent(ticker);
        end
        
        function FrameAnalyzer(self, p, run, fluids, mag)
            figure(1);
            
            c = self.cut;

            colorset={'b','r','g'};            
% FIXME: Add support for 'drawnow' to force graphics updates over e.g. laggy remote connections
            hold off;
            for i = 1:numel(fluids);
                plotdat = fluids(i).mass.array;
                if self.plotDifference; plotdat = plotdat - self.q0{i}; end
                
                switch(self.plotmode)
                    case 1
                        plot(plotdat(:,c(2),c(3) ), colorset{i});
                    case 2
                        plot(squish(plotdat(c(1),:,c(3)) ), colorset{i});
                    case 3
                        plot(squish(plotdat(c(1),c(2),:)), colorset{i});
                    case 4
                        self.mkimage(plotdat(:,:,c(3)));
                    case 5
                        self.mkimage(squish(plotdat(:,c(2),:)));
                    case 6
                        self.mkimage(squish(plotdat(c(1),:,:)));
                    case 7
                        mass = fluids(i).mass; mom = fluids(i).mom; ener = fluids(i).ener;
                        plotem();
                end
                if i == 1; hold on; end
            end
            
            title(sum(run.time.history));
            if self.forceRedraw; drawnow; end
            
            % Rearm myself
            p.iter = p.iter + self.iterationsPerCall;
            p.active = 1;
            
            if self.insertPause; input('Enter to continue: '); end
        end
        
        function finalize(self, run, fluids, mag)
            run.save.logPrint('Realtime Plotter finalized.\n');
        end
    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                      P R O T E C T E D    [M]
        function mkimage(self, data)
            if self.generateTeletextPlots
                ttplot(data);
            else
                imagesc(data);
            end
        end
    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %                                                 S T A T I C    [M]

    end%PROTECTED
    
end%CLASS
