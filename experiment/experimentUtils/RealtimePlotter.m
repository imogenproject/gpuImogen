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
        subsets;

        plotDifference;
        insertPause;
        
        iterationsPerCall;
        firstCallIteration;

        generateTeletextPlots;

        plotmode; % 1 = one, 2 = 2 horizontal, 3 = 2 vertical, 4 = 2x2 matrix
        plotProps;

        spawnGUI;
        forceRedraw;

    end %PUBLIC
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
        pNumFluids; % copied for convenient reference by GUI at init time
        pResolution; % copied for reference by cut setter function

        pSubsX; pSubsY; pSubsZ;

        pGUIFigureNumber;
        pGUISelectedPlotnum;
        pGUIPauseSpin;
    end %PROTECTED
    
    %===================================================================================================
    methods %                                                                     G E T / S E T  [M]
    end%GET/SET
    
    %===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
        
        function self = RealtimePlotter()
            self = self@LinkedListNode(); % init LL data
            
            self.q0                 = [];
            self.insertPause        = 0;
        
            self.iterationsPerCall  = 100;
            self.firstCallIteration = 1;

            self.cut = -[1 1 1];
            self.subsets = -[1 1 1;1 1 1;1 1 1];

            self.generateTeletextPlots = 0; % FIXME: set this depending on availability of graphics
            self.forceRedraw           = 0;
        
            self.plotProps = struct('fluidnum',1,'what',1,'logscale',0,'slice',1,'plottype',1,'grid',0,'cbar',0);
            self.plotProps(2:4) = self.plotProps(1);

            self.spawnGUI           = 0;
            self.pGUIPauseSpin       = 0;

            self.pGUISelectedPlotnum = 1;

            self.plotmode = 1; % default to one plot
        end
        
        function initialize(self, IC, run, fluids, mag)
        % .initialize(IC, run, fluids, mag)
        % IC: initial conditions
        % run: ImogenManager created by Imogen
        % fluids: FluidManager(:) array, with fluids initialized
        % mag: MagnetArray(3)
        % This functionality is NOT compatible with MPI, and will make the run exit if # ranks > 1
            forceStop_ParallelIncompatible();
            
            numFluids = numel(fluids);
            self.pNumFluids = numFluids;
            
            self.pResolution = fluids(1).mass.gridSize;

            for i = 1:numFluids;
                self.q0{i} = fluids(i).mass.array;
            end

            for i=1:3;
                if self.cut(i) < 0; self.cut(i) = ceil(size(self.q0{1},i)/2); end

                if self.subsets(i,1) < 0; self.subsets(i,1) = 1; end % start
                if self.subsets(i,2) < 0; self.subsets(i,2) = 1; end % step
                if self.subsets(i,3) < 0; self.subsets(i,3) = self.pResolution(i); end % end
            end
            self.updateSubsets();
            
            ticker = ImogenEvent([], self.firstCallIteration, [], @self.FrameAnalyzer);
            ticker.active = 1;
            run.attachEvent(ticker);

            % Throw up a window that lets the user interactive muck with a run's visualization
            if self.spawnGUI; self.pGUIFigureNumber = RealtimePlotterGUI(self); end
        end
        
        function activateGUI(self, fignum)
            self.pGUIFigureNumber = fignum;
            % FIXME: set some gui element refs here for callbacks to play games with
        end

        function printCurrentPlotConfig(self)
            rpn = input('Name for RealtimePlotter class: ','s');

            ap = 1;
            switch(self.plotmode); case 1; ap = 1; case 2; ap = 2; case 3; ap = 2; case 4; ap = 4; end

            fprintf('%s.plotmode = %i;\n', rpn, int32(self.plotmode));
	    fprintf('%s.cut = %s\n%s.subsets = %s\n', mat2str(self.cut), mat2str(self.subsets));

            fieldnames={'fluidnum','what','logscale','slice','plottype','grid','cbar'};
            for pltno = 1:ap
                for fname = 1:7;
                    fprintf('%s.plotProps(%i).%s = %i; ', rpn, int32(pltno), fieldnames{fname}, int32(self.plotProps(pltno).(fieldnames{fname})));
                    if fname == 4; fprintf('\n'); end
                end
                fprintf('\n');
            end
        end

        function FrameAnalyzer(self, p, run, fluids, ~)
            fig = figure(1);
            
            c = self.cut;
            run.time.iteration

            nplots = 1;
            switch(self.plotmode) % one/two horizontal/two vertical/2x2 matrix 
                case 1; nplots = 1;
                case 2; nplots = 2;
                case 3; nplots = 2;
                case 4; nplots = 4;
            end

            for plotnum = 1:nplots
                params = self.plotProps(plotnum);

                q = self.fetchPlotQty(fluids(params.fluidnum), params.slice, params.what);

                if params.logscale
                    q = abs(q);
                end

                self.pickSubplot(plotnum, self.plotmode);
                self.drawPlot(q, params);
                obj = findobj('tag','qtylistbox');
                title(obj.String(params.what,:));
            end

            fig.Name = ['Output at iteration ' num2str(run.time.iteration) ', time ' num2str(sum(run.time.history))];
            if self.forceRedraw; drawnow; end

            % Rearm myself
            p.iter = p.iter + self.iterationsPerCall;
            p.active = 1;

            if self.insertPause;
                if self.spawnGUI
                    % spin in a dummy loop so the GUI can respond
                    self.pGUIPauseSpin = 1;
                    btn = findobj('tag','resumebutton');
                    ct = 1;

                    while self.pGUIPauseSpin;
                        pause(.33);
                        ct = ct + 1;

                        if mod(ct,2)
                            btn.BackgroundColor = [.3 .9 .3];
                        else
                            btn.BackgroundColor = [.9 .3 .3];
                        end
                    end

                    btn.BackgroundColor = [94 94 94]/100;
                else
                    input('Enter to continue: ');
                end
            end
        end

        function Q = fetchPlotQty(self, fluid, sliceID, what)
            u = []; v = []; w = [];
            
            dim = fluid.mass.gridSize;

            switch sliceID
                case 1; u = self.pSubsX; v = self.cut(2); w = self.cut(3); % x
                case 2; u = self.cut(1); v = self.pSubsY; w = self.cut(3); % y
                case 3; u = self.cut(1); v = self.cut(2); v = self.pSubsZ; % z
                case 4; u = self.pSubsX; v = self.pSubsY; w = self.cut(3); % xy
                case 5; u = self.pSubsX; v = self.cut(2); w = self.pSubsZ; % xz
                case 6; u = self.cut(1); v = self.pSubsY; w = self.pSubsZ; % yz
            end

            % NOTE NOTE NOTE the values for 'what' are linked to the ordering of the list entries in the GUI quantity-selection
            % NOTE NOTE NOTE box. search RealtimePlotterGUI.m /lis\ =
            switch what;
            case 1; Q = fluid.mass.array(u,v,w); % rho
            case 2; Q = fluid.mom(1).array(u,v,w); % px
            case 3; Q = fluid.mom(2).array(u,v,w); % py
            case 4; Q = fluid.mom(3).array(u,v,w);% pz
            case 5; Q = fluid.mom(1).array(u,v,w)./fluid.mass.array(u,v,w); %vx 
            case 6; Q = fluid.mom(2).array(u,v,w)./fluid.mass.array(u,v,w); %vy
            case 7; Q = fluid.mom(3).array(u,v,w)./fluid.mass.array(u,v,w); %vz
            case 8; Q = fluid.ener.array(u,v,w); %etotal
            case 9; % pressure
	        Q = fluid.calcPressureOnCPU();
		Q = Q(u,v,w);
            case 10;% temperature
	        Q = fluid.calcPressureOnCPU();
		Q = Q(u,v,w)./fluid.mass.array(u,v,w);
            end

            Q = squish(Q); % flatten for return
        end

        function pickSubplot(self, plotnumber, plotmode)
            figure(1);

            switch plotmode
            case 1; % one plot
                subplot(1,1,1);
            case 2; % 2 left-right plots
                if plotnumber == 1; subplot(1,2,1); else; subplot(1,2,2); end
            case 3; % 2 vertical plots
                if plotnumber == 1; subplot(2,1,1); else; subplot(2,1,2); end
            case 4; % 2x2 matrix of plots
                subplot(2,2,plotnumber);
            end
        end

        function drawPlot(self, q, decor)
            if decor.slice < 4; % x/y/z cut: one dimensional: do plot()
                if decor.logscale
                    semilogy(q)
                else
                    plot(q);
                end
                if decor.grid; grid on; end
                if decor.cbar; colorbar; end
            else % plottype: 1 -> imagesc, 2 -> surf
                if decor.logscale; q = log10(q); end

                if decor.plottype == 1
                    imagesc(q);
                else
                    surf(q,'linestyle','none');
                end

                if decor.cbar; colorbar; end
            end
        end
        function oldFrameAnalyzer(self, p, run, fluids, ~)
           figure(1);

            c = self.cut;
            run.time.iteration

            colorset={'b','r','g'};            
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
                        q = plotdat(:,:,c(3));
                        if run.geometry.pGeometryType == ENUM.GEOMETRY_CYLINDRICAL
                           [r, phi] = run.geometry.ndgridSetIJ('pos');
                           u = r(:,1);
                           v = squeeze(phi(1,:));
                           
                           mm = max(u) * 1.05;
                           dmm = .0025*mm;
                           
                           [x, y] = ndgrid(-mm:dmm:mm, -mm:dmm:mm);
                           rquer = sqrt(x.^2+y.^2);
                           phiquer = atan2(y,x);
                           
                           dinterp = interp2(v, u, q, phiquer+pi,rquer);
                           self.mkimage(dinterp);
                        end
                        %self.mkimage(q);
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

    %========== These are the UI callbacks hooked by RealtimePlotterGUI
        function gcbSetPause(self, src, data)
            if src.Value == 1
                self.insertPause = 1;
                src.BackgroundColor = [3 9 3]/10;
                src.String = 'Pause on call';
            else
                self.insertPause = 0;
                src.BackgroundColor = [9 3 3]/10;
                src.String = 'No pause on call';
            end
        end
        function gcbResumeFromSpin(self, src, data)
            self.pGUIPauseSpin = 0;
        end
        function gcbSetRedraw(self, src, data)
            if src.Value == 1
                self.forceRedraw = 1;
                src.BackgroundColor = [3 9 3]/10;
                src.String = 'Force redraw';
            else
                self.forceRedraw = 0;
                src.BackgroundColor = [9 3 3]/10;
                src.String = 'No forced redraw';
            end
        end
        function gcbSetItersPerCall(self, src, data)
            S = str2num(src.String);

            if isempty(S)
                src.String = 'Enter a positive integer';
            else
                if numel(S) > 1; S = S(1); end

                S = round(S);
                if S < 1
                    src.String = 'Enter one positive integer';
                else
                    self.iterationsPerCall = S;
                    src.String = num2str(S);
                end
            end
        end
        function gcbDumpConfiguration(self, src, data)
            self.printCurrentPlotConfig();
        end
        function gcbCycleNumOfPlots(self, src, data)

            self.plotmode = mod(self.plotmode, 4) + 1; %1->2,2->3,3->4,4->1

            switch self.plotmode
                case 1; src.String = 'One plot';
                case 2; src.String = '2 plots horizontally';
                case 3; src.String = '2 plots above eachother';
                case 4; src.String = '2x2 matrix of plots';
            end
        end
        function gcbSetPlotFluidsrc(self, src, data) % called by the --/++ arrows by 'FLUID: N'
            F = self.plotProps(self.pGUISelectedPlotnum).fluidnum;
            if src.value < 0;
                F = F - 1;
                if F < 1; F = 1; end
            end
            if src.value > 0;
                F = F + 1;
                if F > self.pNumFluids; F = self.pNumFluids; end
            end
            self.plotProps(self.pGUISelectedPlotnum).fluidnum = F;

            obj = findobj('tag','fluidnumbertxt');
            obj.String = ['Fluid: ' num2str(F)];
        end
        function gcbChoosePlotQuantity(self, src, data) % called by listplot of qtys to plot
            self.plotProps(self.pGUISelectedPlotnum).what = src.Value;
        end
        function gcbCyclePlotSelection(self, src, data)
            plotsActive = 1;
            switch self.plotmode;
                case 1; plotsActive = 1; case 2; plotsActive = 2; case 3; plotsActive = 2; case 4; plotsActive = 4;
            end

            self.pGUISelectedPlotnum = mod(self.pGUISelectedPlotnum, plotsActive) + 1; % 1->2->3->4->1
            plotno = self.pGUISelectedPlotnum;

            src.String = ['Editing properties of plot ' num2str(plotno)];

            obj = findobj('tag','qtylistbox'); obj.Value = self.plotProps(plotno).what;
            obj = findobj('tag','colorbarbutton'); obj.Value = self.plotProps(plotno).cbar; if obj.Value == 0; obj.BackgroundColor = [94 94 94]/100; else; obj.BackgroundColor = [3 9 3]/10; end
            obj = findobj('tag','gridbutton'); obj.Value = self.plotProps(plotno).grid; if obj.Value == 0; obj.BackgroundColor = [94 94 94]/100; else; obj.BackgroundColor = [3 9 3]/10; end
            obj = findobj('tag','logbutton'); obj.Value = self.plotProps(plotno).logscale; if obj.Value == 0; obj.BackgroundColor = [94 94 94]/100; else; obj.BackgroundColor = [3 9 3]/10; end
            obj = findobj('tag','fluidnumbertxt'); obj.String = ['Fluid: ' num2str(self.plotProps(plotno).fluidnum)];
            obj = findobj('tag','plottypebutton');
            labels = {'imagesc','surf','plot'};
            labno = self.plotProps(plotno).plottype;
            if self.plotProps(plotno).slice < 4; labno = 3; end
            obj.String = labels{labno};
        end
        function gcbCyclePlotmode(self, src, data)
            M = mod(self.plotProps(self.pGUISelectedPlotnum).plottype, 2) + 1;

            if self.plotProps(self.pGUISelectedPlotnum).slice < 4; % 1d output
                src.String = 'plot';
            else
                labels = {'imagesc','surf'};
                src.String = labels{M};
                self.plotProps(self.pGUISelectedPlotnum).plottype = M;
            end
        end
        function gcbToggleColorbar(self, src, data) 
            if src.Value == 1; L = 1; else; L = 0; end
            self.plotProps(self.pGUISelectedPlotnum).cbar = L;
            if L % yes colorbar: green it
                src.BackgroundColor = [3 9 3]/10;
            else
                src.BackgroundColor = [94 94 94]/100;
            end
        end
        function gcbToggleGrid(self, src, data)
            if src.Value == 1; G = 1; else; G = 0; end
            self.plotProps(self.pGUISelectedPlotnum).grid = G;

            if G; % yes grid: green it
                src.BackgroundColor = [3 9 3]/10;
            else
                src.BackgroundColor = [94 94 94]/100;
            end
        end
        function gcbToggleLogScale(self, src, data)
            if src.Value == 1; L = 1; else; L = 0; end
            self.plotProps(self.pGUISelectedPlotnum).logscale = L;

            if L % yes log scale: green button
                src.String = 'log10';
                src.BackgroundColor = [3 9 3]/10;
            else
                src.String = 'linear';
                src.BackgroundColor = [94 94 94]/100;
            end
        end
        function gcbSetSlice(self, src, data) % queue on src.String
            tagnames={'xSliceButton','ySliceButton','zSliceButton','xySliceButton','xzSliceButton','yzSliceButton'};
            whodunit = strcmp(src.Tag, tagnames);

            self.plotProps(self.pGUISelectedPlotnum).slice = find(whodunit);

            if src.Value == 0; src.Value = 1; else
                for n = 1:6; % mark all other buttons off (mutex)
                    if whodunit(n) == 0; element = findobj('Tag',tagnames{n}); element.Value = 0; end
                end
            end

            obj = findobj('tag','plottypebutton');
            if self.plotProps(self.pGUISelectedPlotnum).slice < 4; 
                obj.String='plot';
            else
                labels = {'imagesc','surf'};
                obj.String = labels{self.plotProps(self.pGUISelectedPlotnum).plottype};
            end

        end
        function gcbSetCuts(self, src, data) % queue on src.value to determine what to set in self.cuts()
            % FIXME this needs to be teh implemented

            N = str2num(src.Tag(8:9)); % quick'n'dirty
            val = str2num(src.String);

            val = round(val); if val < 1; val = 1; end % true for all inputs

            switch N
                case 11; if val > self.pResolution(3); val = self.pResolution(3); end; self.cut(3) = val; % z cut
                case 12; if val > self.pResolution(3); val = self.pResolution(3); end; if val > self.subsets(3,3); self.subsets(3,3) = val; end; self.subsets(3,1) = val; 
                case 13; self.subsets(3,2) = val;
                case 14; if val > self.pResolution(3); val = self.pResolution(3); end; if val < self.subsets(3,1); self.subsets(3,1) = val; end; self.subsets(3,3) = val;

                case 21; if val > self.pResolution(2); val = self.pResolution(2); end; self.cut(2) = val; % y cut
                case 22; if val > self.pResolution(2); val = self.pResolution(2); end; if val > self.subsets(2,3); self.subsets(2,3) = val; end; self.subsets(2,1) = val;
                case 23; self.subsets(2,2) = val;
                case 24; if val > self.pResolution(2); val = self.pResolution(2); end; if val < self.subsets(2,1); self.subsets(2,1) = val; end; self.subsets(2,3) = val;
                
                case 31; if val > self.pResolution(1); val = self.pResolution(1); end; self.cut(1) = val; % x cut
                case 32; if val > self.pResolution(1); val = self.pResolution(1); end; if val > self.subsets(1,3); self.subsets(1,3) = val; end; self.subsets(1,1) = val;
                case 33; self.subsets(1,2) = val;
                case 34; if val > self.pResolution(1); val = self.pResolution(1); end; if val < self.subsets(1,1); self.subsets(1,1) = val; end; self.subsets(1,3) = val;
                otherwise; error('Oh wow, much bad, self.gcbSetCuts called with invalid src.Tag: lolwut?');
            end
            self.updateSubsets();
            src.String = num2str(val);
        end

    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                      P R O T E C T E D    [M]
        function updateSubsets(self)
            self.pSubsX = self.subsets(1,1):self.subsets(1,2):self.subsets(1,3);
            self.pSubsY = self.subsets(2,1):self.subsets(2,2):self.subsets(2,3);
            self.pSubsZ = self.subsets(3,1):self.subsets(3,2):self.subsets(3,3);
        end

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
