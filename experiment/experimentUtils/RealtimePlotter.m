classdef RealtimePlotter <  LinkedListNode
% An interactive peripheral for displaying the state of a simulation as
% it runs, in realtime. This is usable only by node-serial simulations, 
% and throws an error to all ranks if run in parallel (simulation fails,
% parImogenLoad moves on to next runfile)
%___________________________________________________________________________________________________
    
%===================================================================================================
    properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        q0;
        cut;
        indSubs;

        plotDifference;
        insertPause;
        
        iterationsPerCall;
        firstCallIteration;

        generateTeletextPlots;

        plotmode; % 1 = one, 2 = 2 horizontal, 3 = 2 vertical, 4 = 2x2 matrix
        plotProps;

        outputMovieFrames;

        spawnGUI;
        forceRedraw;

    end %PUBLIC
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
        pNumFluids; % copied for convenient reference by GUI at init time
        pResolution; % copied for reference by cut setter function

        pSubsX; pSubsY; pSubsZ;
        pCoords;

        pGUIFigureNumber;
        pGUISelectedPlotnum;
        pGUIPauseSpin;
        pGUIPlotsNeedRedraw;
        
        pGeometryMgrHandle;

        pDisplayedPlotOffset;

        pstatic_colorchars = 'rgbcmykw';
        pstatic_ppfields = {'fluidnum','what','logscale','slice','plottype','grid','cbar','axmode','velvecs','vv_scale','vv_decfac','vv_weight','vv_color','vv_type'};

        pAxisTypeLabels = {'axis off','cell #','pixels','position'};

        pMovieNextFrame;
	pMovieFramePrefix;

	pCAct =  [9 3 3]/10;
	pCEnab = [3 9 3]/10;
	pCNeut = [94 94 94]/100;
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
            self.indSubs = -[1 1 1;1 1 1;1 1 1];

            self.generateTeletextPlots = 0; % FIXME: set this depending on availability of graphics
            self.forceRedraw           = 0;
        
            self.plotProps = struct('fluidnum',1,'what',1,'logscale',0,'slice',1,'plottype',1,'grid',0,'cbar',0,'axmode',0,'velvecs',0,'vv_scale',1,'vv_decfac',10,'vv_weight',1,'vv_color',8,'vv_type',1);
            self.plotProps(2:4) = self.plotProps(1);

            self.spawnGUI           = 0;
            self.pGUIPauseSpin       = 0;

            self.pGUISelectedPlotnum = 1;
            self.pGUIPlotsNeedRedraw = 0;

            self.pDisplayedPlotOffset = 0; 

            self.outputMovieFrames = 0;
            self.pMovieNextFrame = 0;
	    self.pMovieFramePrefix = 'RTP_';

            self.plotmode = 1; % default to one plot
        end

        function vectorToPlotprops(self, id, v)
            if nargin < 3
                whos
                error('Did not receive correct number of args: RealtimePlotter.vectorToPlotprops(index 1...4, vector);');
            end
            if numel(v) ~= 14
                whos
                error('Attempt to use RealtimePlotter.vectorToPlotprops() but input numeric vector does not have 14 elements.');
            end

            fields = self.pstatic_ppfields;
            for k = 1:numel(fields)
                self.plotProps(id).(fields{k}) = v(k);
            end
        end

	function movieProps(self, active, nxt, prefix)
	    self.outputMovieFrames = active;
	    self.pMovieNextFrame = nxt;
	    self.pMovieFramePrefix = prefix;
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

                if self.indSubs(i,1) < 0; self.indSubs(i,1) = 1; end % start
                if self.indSubs(i,2) < 0; self.indSubs(i,2) = 1; end % step
                if self.indSubs(i,3) < 0; self.indSubs(i,3) = self.pResolution(i); end % end
            end

            self.pGeometryMgrHandle = run.geometry;
            self.updateSubsets();
            
            ticker = ImogenEvent([], self.firstCallIteration, [], @self.FrameAnalyzer);
            ticker.active = 1;
            run.attachEvent(ticker);

            % Throw up a window that lets the user interactive muck with a run's visualization
            if self.spawnGUI; self.pGUIFigureNumber = RealtimePlotterGUI(self); end
        end

        function destroy(self)
            self.closeGUI();
        end

        function activateGUI(self, fignum)
            self.pGUIFigureNumber = fignum;
            % FIXME: set some gui element refs here for callbacks to play games with
        end
        function closeGUI(self)
            close(32);
        end

        function printCurrentPlotConfig(self)
            rpn = input('Name for RealtimePlotter class: ','s');

            outtype = input('enter 1 for human-readable, 2 for machine code: ');

            ap = 1;
            switch(self.plotmode); case 1; ap = 1; case 2; ap = 2; case 3; ap = 2; case 4; ap = 4; end

            fprintf('%s.plotmode = %i;\n', rpn, int32(self.plotmode));
            fprintf('%s.cut = %s\n%s.indSubs = %s;\n', rpn, mat2str(self.cut), rpn, mat2str(self.indSubs));
	    fprintf('%s.movieProps(%i, %i, ''%s'');\n', rpn, int32(self.outputMovieFrames), int32(self.pMovieNextFrame), self.pMovieFramePrefix);

            if outtype == 1
                fieldnames = self.pstatic_ppfields;
                for pltno = 1:ap
                    for fname = 1:numel(fieldnames);
                        fprintf('%s.plotProps(%i).%s = %i; ', rpn, int32(pltno), fieldnames{fname}, int32(self.plotProps(pltno).(fieldnames{fname})));
                        if mod(fname, 4) == 0; fprintf('\n'); end
                    end
                    fprintf('\n');
                end
            else
                fieldnames = self.pstatic_ppfields;
                for pltno = 1:ap
                    psv = [];
                    for fname = 1:numel(fieldnames)
                        psv(end+1) = self.plotProps(pltno).(fieldnames{fname});
                    end
                    fprintf('%s.vectorToPlotprops(%i, [%s]);\n', rpn, int32(pltno), num2str(psv));
                end
            end
        end

        function drawGfx(self, run, fluids)
            fig = figure(1);

            nplots = 1;
            switch(self.plotmode) % one/two horizontal/two vertical/2x2 matrix 
                case 1; nplots = 1;
                case 2; nplots = 2;
                case 3; nplots = 2;
                case 4; nplots = 4;
            end

            for plotnum = 1:nplots
                plotid = mod(plotnum + self.pDisplayedPlotOffset-1, 4) + 1;

                params = self.plotProps(plotid);

                q = self.fetchPlotQty(fluids(params.fluidnum), params.slice, params.what);

                if params.logscale
                    q = abs(q);
                end
                
                if params.velvecs == 1
                    switch(params.slice)
                        case 4; vv = self.fetchPlotQty(fluids(params.fluidnum), params.slice, 98 + 3*params.vv_type);
                        case 5; vv = self.fetchPlotQty(fluids(params.fluidnum), params.slice, 99 + 3*params.vv_type);
                        case 6; vv = self.fetchPlotQty(fluids(params.fluidnum), params.slice, 100 + 3*params.vv_type);
                    end
                else
                    vv = [];
                end

                self.pickSubplot(plotnum, self.plotmode);
                self.drawPlot(q, params, vv);
                obj = findobj('tag','qtylistbox');
                title(obj.String(params.what,:));
            end

            fig.Name = ['Output at iteration ' num2str(run.time.iteration) ', time ' num2str(sum(run.time.history))];
            if self.forceRedraw; drawnow; end

        end
        
        function FrameAnalyzer(self, p, run, fluids, ~)
            self.drawGfx(run, fluids);
            
            if self.insertPause;
                if self.spawnGUI
                    % spin in a dummy loop so the GUI can respond
                    self.pGUIPauseSpin = 1;
                    btn = findobj('tag','resumebutton');
                    ct = 1;

                    while self.pGUIPauseSpin;
                        pause(.33);
                        if self.pGUIPlotsNeedRedraw
                            self.drawGfx(run, fluids)
                            self.pGUIPlotsNeedRedraw = 0;
                        end
                        ct = ct + 1;

                        if mod(ct,2)
                            btn.BackgroundColor = [.3 .9 .3];
                        else
                            btn.BackgroundColor = [.9 .3 .3];
                        end
                    end

                    btn.BackgroundColor = self.pCNeut;
                else
                    input('Enter to continue: ');
                end

            end

            if self.outputMovieFrames
                self.emitMovieFrame(run.paths.image, self.pMovieFramePrefix);
            end

            % Rearm myself
            p.iter = p.iter + self.iterationsPerCall;
            p.active = 1;
        end

        function emitMovieFrame(self, path, prefix)
            f = figure(1);
            imgdat = getframe(f);
            imwrite(imgdat.cdata, sprintf('%s/%s%05i.png', path, prefix, int32(self.pMovieNextFrame)));
        
            self.pMovieNextFrame = self.pMovieNextFrame+1;
	    x = findobj('tag','movieframereport');
	    x.String = sprintf('Next frame: %i', int32(self.pMovieNextFrame));
        end

        function Q = fetchPlotQty(self, fluid, sliceID, what)
            u = []; v = []; w = [];
            
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
            % LARGE VALUES SPECIFY OTHER THINGS
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
            case 101; % XY velocity
                Q = { fluid.mom(1).array(u,v,w)./fluid.mass.array(u,v,w), fluid.mom(2).array(u,v,w)./fluid.mass.array(u,v,w) };
            case 102; % XZ velocity
                Q = { fluid.mom(1).array(u,v,w)./fluid.mass.array(u,v,w), fluid.mom(3).array(u,v,w)./fluid.mass.array(u,v,w) };
            case 103; % YZ velocity
                Q = { fluid.mom(2).array(u,v,w)./fluid.mass.array(u,v,w), fluid.mom(3).array(u,v,w)./fluid.mass.array(u,v,w) };
            case 104; % 
                QQ = comovingAcceleration(fluid, fluid.parent.potentialField.field.array, fluid.parent.geometry, fluid.parent.frameTracking);
                Q = {QQ{1}(u,v,w), QQ{2}(u,v,w)};
            case 105;
                QQ = comovingAcceleration(fluid, fluid.parent.potentialField.field.array, fluid.parent.geometry, fluid.parent.frameTracking);
                Q = {QQ{1}(u,v,w), QQ{3}(u,v,w)};
            case 106;
                QQ = comovingAcceleration(fluid, fluid.parent.potentialField.field.array, fluid.parent.geometry, fluid.parent.frameTracking);
                Q = {QQ{2}(u,v,w), QQ{3}(u,v,w)};
            default;
                error(['Fatal: received value of ' num2str(what) ' that is unhandled.']);
            end

            Q = squish(Q); % flatten for return
        end

        function pickSubplot(self, plotnumber, plotmode)
            figure(1);

            switch plotmode
            case 1; % one plot
                subplot(1,1,1);
            case 2; % 2 left-right plots
                if plotnumber == 1; subplot(1,2,1); else subplot(1,2,2); end
            case 3; % 2 vertical plots
                if plotnumber == 1; subplot(2,1,1); else subplot(2,1,2); end
            case 4; % 2x2 matrix of plots
                subplot(2,2,plotnumber);
            end
        end

        function drawPlot(self, q, decor, velocityVectors)
            if decor.slice < 4; % x/y/z cut: one dimensional: do plot()
                axval = self.pCoords{decor.slice};
                % axmode = 0 -> off, 1 -> px, 2 -> cell #, 3 -> position
                if decor.axmode == 1; axval = 1:numel(axval); end
                if decor.axmode == 2; axval = -axval; end
                % case 3 is the default
                if decor.logscale
                    semilogy(axval, q)
                else
                    plot(axval, q);
                end
                if decor.axmode == 0; axis off; else; axis on; end
                if decor.grid; grid on; end
                if decor.cbar; colorbar; end
            else % plottype: 1 -> imagesc, 2 -> surf
                if decor.logscale; q = log10(q); end

                axh = []; axv = [];
                switch(decor.slice)
                    case 4; axh = self.pCoords{2}; axv = self.pCoords{1};
                    case 5; axh = self.pCoords{3}; axv = self.pCoords{1};
                    case 6; axh = self.pCoords{3}; axv = self.pCoords{2};
                end
                
                if decor.axmode == 1; axh = 1:numel(axh); axv = 1:numel(axv); end
                if decor.axmode == 2; axh = -axh; axv = -axv; end

                if decor.plottype == 1
                    imagesc(axh, axv, q);
                else
                    surf(axh, axv, q,'linestyle','none');
                end
                
                if decor.velvecs == 1
                    vvx = squeeze(velocityVectors{1});
                    vvy = squeeze(velocityVectors{2});
                    
                    % load the decimation factor, arrow scale factor, line weight and color for the vector field plots
                    df = decor.vv_decfac;
                    vfScale = decor.vv_scale;
                    lw = decor.vv_weight;
                    colorChar = self.pstatic_colorchars(decor.vv_color);

                    % And throw it up onto the plot
                    hold on;
                    quiver(axh(1:df:end), axv(1:df:end), vvy(1:df:end,1:df:end), vvx(1:df:end,1:df:end),vfScale,colorChar, 'LineWidth',lw)
                    hold off;
                end
                
                if decor.axmode == 0; axis off; else; axis on; end
                
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
                    ase 4
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
            if self.spawnGUI; % close the gui control window
                f = findobj('tag','ImogenRTP_GUIWindow');
                if ~isempty(f); close(f); end
            end
            run.save.logPrint('Realtime Plotter finalized.\n');
        end

    %========== These are the UI callbacks hooked by RealtimePlotterGUI
        function gcbSetPause(self, src, data)
            if src.Value == 1
                self.insertPause = 1;
                src.BackgroundColor = self.pCEnab;
                src.String = 'Pause on call';
            else
                self.insertPause = 0;
                src.BackgroundColor = self.pCAct;
                src.String = 'No pause on call';
            end
        end
        function gcbResumeFromSpin(self, src, data)
            self.pGUIPauseSpin = 0;
        end
        function gcbSetRedraw(self, src, data)
            if src.Value == 1
                self.forceRedraw = 1;
                src.BackgroundColor = self.pCEnab;
                src.String = 'Force redraw';
            else
                self.forceRedraw = 0;
                src.BackgroundColor = self.pCAct;
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
            self.pGUIPlotsNeedRedraw = 1;
        end
        function gcbCyclePlotOffset(self, src, data)
            self.pDisplayedPlotOffset  = mod(self.pDisplayedPlotOffset + 1, 4); % 0,1,2,3,0,...
            self.pGUIPlotsNeedRedraw = 1;
        end
        function gcbSetPlotFluidsrc(self, src, data) % called by the --/++ arrows by 'FLUID: N'
            F = self.plotProps(self.pGUISelectedPlotnum).fluidnum;
            if src.Value < 0;
                F = F - 1;
                if F < 1; F = 1; end
            end
            if src.Value > 0;
                F = F + 1;
                if F > self.pNumFluids; F = self.pNumFluids; end
            end
            self.plotProps(self.pGUISelectedPlotnum).fluidnum = F;

            obj = findobj('tag','fluidnumbertxt');
            obj.String = ['Fluid: ' num2str(F)];
            self.pGUIPlotsNeedRedraw = 1;
        end
        function gcbChoosePlotQuantity(self, src, data) % called by listplot of qtys to plot
            self.plotProps(self.pGUISelectedPlotnum).what = src.Value;
            self.pGUIPlotsNeedRedraw = 1;
        end
        function gcbCyclePlotSelection(self, src, data)
            plotsActive = 1;
            switch self.plotmode;
                case 1; plotsActive = 1; case 2; plotsActive = 2; case 3; plotsActive = 2; case 4; plotsActive = 4;
            end

            self.pGUISelectedPlotnum = mod(self.pGUISelectedPlotnum, plotsActive) + 1; % 1->2->3->4->1
            plotno = self.pGUISelectedPlotnum;

            src.String = ['Editing plot ' num2str(plotno)];

	    self.refreshButtonStates();

            self.pGUIPlotsNeedRedraw = 1;
        end
        
        function gcbToggleVelocityField(self, src, data)
            if src.Value == 1; F = 1; else; F = 0; end
            
            self.plotProps(self.pGUISelectedPlotnum).velvecs = F;
            
            if F
                src.BackgroundColor = self.pCEnab;
            else
                src.BackgroundColor = self.pCNeut;
            end
            self.pGUIPlotsNeedRedraw = 1;
            
        end
        function gcbCycleAxisLabels(self, src, data)
            a = mod(self.plotProps(self.pGUISelectedPlotnum).axmode+1,4);
            self.plotProps(self.pGUISelectedPlotnum).axmode = a;
            
            obj = findobj('tag','axeslabelsbutton');
	    obj.String = self.pAxisTypeLabels{a+1};
            
            self.pGUIPlotsNeedRedraw = 1;
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
            self.pGUIPlotsNeedRedraw = 1;
        end
        function gcbToggleColorbar(self, src, data) 
            if src.Value == 1; L = 1; else; L = 0; end
            self.plotProps(self.pGUISelectedPlotnum).cbar = L;
            if L % yes colorbar: green it
                src.BackgroundColor = self.pCEnab;
            else
                src.BackgroundColor = self.pCNeut;
            end
            self.pGUIPlotsNeedRedraw = 1;
        end
        function gcbToggleGrid(self, src, data)
            if src.Value == 1; G = 1; else; G = 0; end
            self.plotProps(self.pGUISelectedPlotnum).grid = G;

            if G; % yes grid: green it
                src.BackgroundColor = self.pCEnab;
            else
                src.BackgroundColor = self.pCNeut;
            end
            self.pGUIPlotsNeedRedraw = 1;
        end
        function gcbToggleLogScale(self, src, data)
            if src.Value == 1; L = 1; else; L = 0; end
            self.plotProps(self.pGUISelectedPlotnum).logscale = L;

            if L % yes log scale: green button
                src.String = 'log10';
                src.BackgroundColor = self.pCEnab;
            else
                src.String = 'linear';
                src.BackgroundColor = self.pCNeut;
            end
            self.pGUIPlotsNeedRedraw = 1;
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
            self.pGUIPlotsNeedRedraw = 1;
        end
        function gcbSetCuts(self, src, data) % queue on src.value to determine what to set in self.cuts()
            N = str2double(src.Tag(8:9)); % quick'n'dirty
         
            val = str2double(src.String);
            if (isfinite(val) == 0) || (isreal(val) == 0); src.String=':('; return; end;
            val = round(val); if val < 1; val = 1; end % true for all inputs

            switch N
                case 11; val = self.clampValue(val, 1, self.pResolution(3), 1);   self.cut(3) = val; % z cut
                case 12; val = self.clampValue(val, 1, self.pResolution(3), 1);   if val > self.indSubs(3,3); self.indSubs(3,3) = val; end; self.indSubs(3,1) = val; 
                case 13; val = self.clampValue(val, 1, self.pResolution(3)-1, 1); self.indSubs(3,2) = val;
                case 14; val = self.clampValue(val, 1, self.pResolution(3), 1);   if val < self.indSubs(3,1); self.indSubs(3,1) = val; end; self.indSubs(3,3) = val;

                case 21; val = self.clampValue(val, 1, self.pResolution(2), 1);   self.cut(2) = val; % y cut
                case 22; val = self.clampValue(val, 1, self.pResolution(2), 1);   if val > self.indSubs(2,3); self.indSubs(2,3) = val; end; self.indSubs(2,1) = val; 
                case 23; val = self.clampValue(val, 1, self.pResolution(2)-1, 1); self.indSubs(2,2) = val;
                case 24; val = self.clampValue(val, 1, self.pResolution(2), 1);   if val < self.indSubs(2,1); self.indSubs(2,1) = val; end; self.indSubs(2,3) = val;
                    
                case 31; val = self.clampValue(val, 1, self.pResolution(1), 1);   self.cut(1) = val; % y cut
                case 32; val = self.clampValue(val, 1, self.pResolution(1), 1);   if val > self.indSubs(1,3); self.indSubs(1,3) = val; end; self.indSubs(1,1) = val; 
                case 33; val = self.clampValue(val, 1, self.pResolution(1)-1, 1); self.indSubs(1,2) = val;
                case 34; val = self.clampValue(val, 1, self.pResolution(1), 1);   if val < self.indSubs(1,1); self.indSubs(1,1) = val; end; self.indSubs(1,3) = val;

                otherwise; error('Much bad, self.gcbSetCuts called with invalid src.Tag. Crash time.');
            end
            self.updateSubsets();
            src.String = num2str(val);
            self.pGUIPlotsNeedRedraw = 1;
        end

        function gcbSetVF_df(self, src, data)
            S = str2num(src.String);

            if isempty(S)
                src.String = 'Enter a positive integer';
            else
                if numel(S) > 1; S = S(1); end

                S = round(S);
                if S < 1
                    src.String = 'Enter one positive integer';
                else
                    self.plotProps(self.pGUISelectedPlotnum).vv_decfac = S;
                    src.String = num2str(S);
                end
            end
            self.pGUIPlotsNeedRedraw = 1;
        end

        function gcbCycleVF_color(self, src, data)
            M = mod(self.plotProps(self.pGUISelectedPlotnum).vv_color, 8) + 1;

            self.plotProps(self.pGUISelectedPlotnum).vv_color = M;

            clrnames = {'Red','Green','Blue','Cyan','Magenta','Yellow','Black','White'};
            src.String = clrnames{M};
            self.pGUIPlotsNeedRedraw = 1;
        end

        function gcbSetVF_weight(self, src, data)
            currentWt = self.plotProps(self.pGUISelectedPlotnum).vv_weight;

            if strcmp(src.Tag,'vf_heavybutton')
                newWt = currentWt + .5;
            else
                newWt = currentWt - .5;
            end

            if newWt < 1; newWt = 1; end

            self.plotProps(self.pGUISelectedPlotnum).vv_weight = newWt;
            if newWt ~= currentWt; self.pGUIPlotsNeedRedraw = 1; end
        end

        function gcbCycleVF_type(self, src, data)
            nutype = mod(self.plotProps(self.pGUISelectedPlotnum).vv_type,2)+1;
            
            switch(nutype)
                case 1; src.String = 'Velocity'; 
                case 2; src.String = 'A_comoving';
            end
            
            self.plotProps(self.pGUISelectedPlotnum).vv_type = nutype;
            self.pGUIPlotsNeedRedraw = 1;
        end

	function gcbMovieSetFrame(self, src, data)
	    S = str2num(src.String);

            if isempty(S)
                src.String = 'Enter a nonnegative integer';
            else
                if numel(S) > 1; S = S(1); end

                S = round(S);
                if S < 0
                    src.String = 'Enter one nonnegative integer';
                else
                    self.pMovieNextFrame = S;
                    src.String = num2str(S);
		    x = findobj('tag','movieframereport');
		    x.String = sprintf('Next frame: %i', int32(S));
                end
            end

	end

	function gcbMoviePrefix(self, src, data)
            S = src.String;

	    % FIXME deblank & sanity check here...

            if isempty(S)
                src.String = 'Enter a nonempty string';
            else
	        self.pMovieFramePrefix = S;
		src.String = S;
            end
	end

	function gcbMovieToggle(self, src, data)
            if src.Value == 1
                self.outputMovieFrames = 1;
                src.BackgroundColor = [9 2 2]/10;
                src.String = 'Writing movie frames';
            else
                self.outputMovieFrames = 0;
                src.BackgroundColor = [9 9 9]/10;
                src.String = 'Not writing frames';
            end

	end
    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                      P R O T E C T E D    [M]
        function updateSubsets(self)
            self.pSubsX = self.indSubs(1,1):self.indSubs(1,2):self.indSubs(1,3);
            self.pSubsY = self.indSubs(2,1):self.indSubs(2,2):self.indSubs(2,3);
            self.pSubsZ = self.indSubs(3,1):self.indSubs(3,2):self.indSubs(3,3);

            if self.pGeometryMgrHandle.pGeometryType == ENUM.GEOMETRY_SQUARE
                self.pCoords{1} = self.pGeometryMgrHandle.localXposition(self.pSubsX);
                self.pCoords{2} = self.pGeometryMgrHandle.localYposition(self.pSubsY);
            else
                self.pCoords{1} = self.pGeometryMgrHandle.localRposition(self.pSubsX);
                self.pCoords{2} = self.pGeometryMgrHandle.localPhiPosition(self.pSubsY);
            end
            self.pCoords{3} = self.pGeometryMgrHandle.localZposition(self.pSubsZ);
        end

	function refreshButtonStates(self)
	    q = self.pGUISelectedPlotnum;
	    pp = self.plotProps(q);

	    % vector field?
	    x = findobj('tag','velfieldbutton');
	    if pp.velvecs; x.Value = 1; x.BackgroundColor = self.PCEnab; else; x.Value = 0; x.BackgroundColor = self.pCNeut; end;

	    % plot qty box
	    x = findobj('tag','qtylistbox');
	    x.Value = pp.what;

	    % axis status
	    x = findobj('tag','axeslabelsbutton');
	    x.String = self.pAxisTypeLabels{pp.axmode+1};

	    % image/surf/plot selection
	    x = findobj('tag','plottypebutton');
	    M = mod(pp.plottype, 2)+1;
	    if pp.slice < 4; % 1d output
                x.String = 'plot';
            else
                labels = {'imagesc','surf'};
                x.String = labels{M};
            end

	    % colorbar
	    x = findobj('tag','colorbarbutton');
	    if pp.cbar == 1; x.Value = 1; x.BackgroundColor = self.pCEnab; else; x.Value = 0; x.BackgroundColor = self.pCNeut; end;

	    % grid
	    x = findobj('tag','gridbutton');
	    if pp.grid == 1; x.Value = 1; x.BackgroundColor = self.pCEnab; else; x.Value = 0; x.BackgroundColor = self.pCNeut; end

	    % log/lin
	    x = findobj('tag','logbutton');
	    if pp.logscale == 1; x.Value = 1; x.BackgroundColor = self.pCEnab; else; x.Value = 0; x.BackgroundColor = self.pCNeut; end

	    %downsample 
	    x = findobj('tag','decfactorbox');
	    x.String = num2str(pp.vv_decfac);

	    % active slice
	    tagnames={'xSliceButton','ySliceButton','zSliceButton','xySliceButton','xzSliceButton','yzSliceButton'};

	    for n = 1:numel(tagnames)
	         x = findobj('tag',tagnames{n});
		 x.Value = (n == pp.slice)*1.0;
	    end

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

        function y = clampValue(x, minval, maxval, mkint)
            y = x;
            if y < minval; y = minval; end
            if y > maxval; y = maxval; end
            if mkint; y = round(y); end
        end

    end%PROTECTED
    
end%CLASS
