classdef RHD_Analyzer < handle
    % Class annotation template for creating new classes.
    %___________________________________________________________________________________________________
    
    %===================================================================================================
    properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        figWindow;
        graphAxes;
        
        ctrlPanel;
        ctrlButtons;
        msgBox;
        dynrngButtons;
        toolsPanel, statusTab, queryTab, statusTxt;
        infoPanel; infoText;
        queryButtons;
        pltGamma;
        
        F; % the data (F)rame, with a nice short variable name :)
        
        runParameters;
        
        xNormalization, xVec;
        tNormalization, timeVec;
        radNormalization;
        shockPos; coldPos;
        fallbackBoost, vfallback;
        
        autovars;
        
        oscilPoints;
        projectedPts;
        
        fOscillate;
        
        fftPoints;
        fftFundamentalFreq;
        
        disableAutoOscil;
        
        datablock;
    end %PUBLIC
    
    properties(Dependent = true)
        runAnalysisPhase;
        automaticMode;
    end
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
        pDPI;
        pCharW, pCharH, pLineH;
        
        pCurrentAnalysisPhase; % 0 = idle, 1 = load, 2= basics, 3 = oscil period, 4 = fft, 5 = commit
        pDisableSequenceButtons;
        pHaveAutovars;
        
        pAbortedSequence;
        
        pAutomaticMode;
        
        pTryAutoOscil;
    end %PROTECTED
    
    %===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        function set.runAnalysisPhase(self, p)
            self.pCurrentAnalysisPhase = p;
            
            bvals = [0 0 0 0 0 0];
            if p == 0; bvals(1) = 0; else
                bvals(1) = 1;
                bvals(p+1) = 1;
            end
            
            for j = 1:numel(bvals); self.ctrlButtons{j}.Value = bvals(j); end
            
            if self.pCurrentAnalysisPhase == 3
                for j = 1:2; self.dynrngButtons{j}.Visible = 1; end
            else
                for j = 1:2; self.dynrngButtons{j}.Visible = 0; end
            end
            
            drawnow();
            
            self.executeCurrentAnalysisPhase();
            
            for j = 1:numel(bvals); self.ctrlButtons{j}.Value = 0; end
            drawnow();
            
            self.pCurrentAnalysisPhase = 0;
            self.disableAutoOscil = 0;
        end
        
        function p = get.runAnalysisPhase(self)
            p = self.pCurrentAnalysisPhase;
        end
    
        function set.automaticMode(self, m)
            if m
                self.pAutomaticMode = m;
                warndlg({'RHD_Analyzer automatic mode is being enabled','All analyses will be automatically inserted','CONVERGENCE LEVEL 5 WILL BE ASSUMED','THIS CAN POTENTIALLY CORRUPT DATA','There will be no further warnings.'}, 'Automatic mode warning');
            else
                self.pAutomaticMode = 0;
            end
        end
        
        function p = get.automaticMode(self)
            p = self.pAutomaticMode;
        end
        
    end%GET/SET
    
    %===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
        
        function dbgclear(self)
            self.pCurrentAnalysisPhase = 0;
            self.runAnalysisPhase = 0;
            self.pTryAutoOscil = 0; 
        end
        
        function dumpContents(self)
            self.F = [];
            self.runParameters = [];
            self.xNormalization = []; self.xVec = [];
            self.tNormalization = []; self.timeVec = [];
            
            self.shockPos = []; self.coldPos = [];
            self.fallbackBoost = 0; self.vfallback = [];
            self.autovars = [];
            self.oscilPoints = [];
            self.projectedPts = [];
            self.fOscillate = [];
            self.fftPoints = [];
            self.fftFundamentalFreq = [];
            self.datablock = [];
        end
        
        function self = RHD_Analyzer()
            self.figWindow = [];
            
            self.interrogateSystemDPI()
            
            figs = findobj('type','figure');
            if numel(figs) == 0
                self.figWindow = figure();
            else
                dlgstring = 'Figure # to attach to?';
                while 1
                    w = inputdlg(dlgstring,'figure selection', [1 40], {num2str(figs(1).Number)});
                    w = str2double(w);
                    for f0 = 1:numel(figs); if figs(f0).Number == w; break; end; end

                    if ~isempty(f0); break; else; dlgstring = 'Invalid figure #. Figure #?'; end
                end
                self.figWindow = figs(f0);
            end
            
            self.figWindow.SizeChangedFcn = @self.handleAnalyzerResize;
            
            self.setupControlPanel();
            
            self.automaticMode         = 0;
            self.pCurrentAnalysisPhase = 0;
            self.pAbortedSequence      = 0;
            self.pTryAutoOscil = 0;
            
            % Buttons: [run full sequence] [auto = 2] [pick oscil period] [pick fft range] [fft] [save] 
            
            self.graphAxes = gca();

            qq.Position = self.figWindow.OuterPosition;
            self.handleAnalyzerResize(qq); 
        end
        
        function setupControlPanel(self)
            dx = self.pCharW;
            
            self.ctrlPanel = uipanel(self.figWindow,'Title','Analysis tools', 'units', 'pixels', 'position', [8, 8, 60*self.pCharW, 4*self.pLineH]);
            
            self.ctrlButtons{1} = uicontrol(self.ctrlPanel, 'Style','togglebutton','String','RUN','tag','runbutton','position',[8,      8, 6*dx, 1.5*self.pCharH], 'callback', @self.cbRunButton);
            
            self.ctrlButtons{2} = uicontrol(self.ctrlPanel, 'Style','togglebutton','String','LOAD','tag','loadbutton','position',[8+7*dx, 8, 6*dx, 1.5*self.pCharH], 'callback', @self.cbSequenceBtns);
            self.ctrlButtons{3} = uicontrol(self.ctrlPanel, 'Style','togglebutton','String','BASIC','tag','basicsbutton','position',[8+14*dx, 8, 6*dx, 1.5*self.pCharH], 'callback', @self.cbSequenceBtns);
            self.ctrlButtons{4} = uicontrol(self.ctrlPanel, 'Style','togglebutton','String','OSCIL','tag','oscilbutton','position',[8+21*dx, 8, 6*dx, 1.5*self.pCharH], 'callback', @self.cbSequenceBtns);
            self.ctrlButtons{5} = uicontrol(self.ctrlPanel, 'Style','togglebutton','String','RANGE','tag','rangebutton','position',[8+28*dx, 8, 6*dx, 1.5*self.pCharH], 'callback', @self.cbSequenceBtns);
            self.ctrlButtons{6} = uicontrol(self.ctrlPanel, 'Style','togglebutton','String','FFT','tag','fftbutton','position',[8+35*dx, 8, 6*dx, 1.5*self.pCharH], 'callback', @self.cbSequenceBtns);
            self.ctrlButtons{7} = uicontrol(self.ctrlPanel, 'Style','togglebutton','String','WRITE','tag','writebutton','position',[8+42*dx, 8, 6*dx, 1.5*self.pCharH], 'callback', @self.cbSequenceBtns);
            self.ctrlButtons{8} = uicontrol(self.ctrlPanel, 'Style','togglebutton','String','TOOLS','tag','statusbutton','position',[8, 8+2*self.pCharH, 6*dx, 1.5*self.pCharH], 'callback', @self.cbShowStatus);
            self.msgBox         = uicontrol(self.ctrlPanel, 'Style','text','String','info', 'position',[8+7*dx, 8+2*self.pCharH, 42*dx, 1.2*self.pCharH]);
            
            self.dynrngButtons{1} = uicontrol(self.figWindow, 'Style','pushbutton','String','Rng x5','tag','dynrngbtn','position',[8, 250+2*self.pCharH, 6*dx, 1.5*self.pCharH], 'callback', @self.cbDynrngPlus);
            self.dynrngButtons{2} = uicontrol(self.figWindow, 'Style','pushbutton','String','Rng /5','tag','dynrngbtn','position',[8, 250, 6*dx, 1.5*self.pCharH], 'callback', @self.cbDynrngMinus);
            
            for q = 1:2; self.dynrngButtons{q}.Visible = 0; end
            
            self.toolsPanel = uitabgroup(self.figWindow,'units', 'pixels', 'position', [8 16+4*self.pLineH, 320, 384], 'visible', 'off');
            
            self.statusTab = uitab(self.toolsPanel,'Title','Status');
            %self.statusTxt = uicontrol('Style','text','Position',[8 8 304 372]);
            self.queryTab  = uitab(self.toolsPanel,'Title','Query');
            self.infoPanel = uipanel(self.statusTab, 'units', 'pixels', 'position', [8 16+4*self.pLineH, 320, 384]);
            %self.infoTab = 
            
            % 16 lines text output
            self.infoText = uicontrol(self.statusTab,'Style','text', 'position', [8 8 306 334], 'string', cell([16 1]),'HorizontalAlignment','left');
            
            self.queryButtons{1} = uicontrol(self.queryTab, 'Style','pushbutton','String','freq','tag','plotfreq','position',   [8, 8              , 8*dx, 1.5*self.pCharH], 'callback',@self.queryPlotCallback);
            self.queryButtons{2} = uicontrol(self.queryTab, 'Style','pushbutton','String','\delta x','tag','plotdx','position', [8, 8+2*self.pCharH, 8*dx, 1.5*self.pCharH], 'callback',@self.queryPlotCallback);
            self.queryButtons{3} = uicontrol(self.queryTab, 'Style','pushbutton','String','\delta L','tag','plotlum','position',[8, 8+4*self.pCharH, 8*dx, 1.5*self.pCharH], 'callback',@self.queryPlotCallback);
            
            self.pltGamma = '53';
            self.queryButtons{4} = uicontrol(self.queryTab, 'Style','pushbutton','String','gam=5/3','tag','gam53','position', [8 + 9*self.pCharW, 8              , 8*dx, 1.5*self.pCharH], 'callback',@self.queryPlotCallback);
            self.queryButtons{5} = uicontrol(self.queryTab, 'Style','pushbutton','String','gam=7/5','tag','gam75','position', [8 + 9*self.pCharW, 8+2*self.pCharH, 8*dx, 1.5*self.pCharH], 'callback',@self.queryPlotCallback);
            self.queryButtons{6} = uicontrol(self.queryTab, 'Style','pushbutton','String','gam=9/7','tag','gam97','position', [8 + 9*self.pCharW, 8+4*self.pCharH, 8*dx, 1.5*self.pCharH], 'callback',@self.queryPlotCallback);
            
            self.queryButtons{7} = uicontrol(self.queryTab, 'Style','togglebutton','String','Enable point query','tag','ptquery','position',[8, 6*self.pCharH, 17*self.pCharW, 1.5*self.pCharH], 'callback',@self.queryPlotCallback);
            self.queryButtons{8} = uicontrol(self.queryTab, 'Style','togglebutton','String','Enable plist gen','tag','plistgen','position',[8, 8*self.pCharH, 17*self.pCharW, 1.5*self.pCharH], 'callback',@self.queryPlotCallback);
            
        end
        
        function updateInfoPanel(self)
            
            if ~isempty(self.runParameters)
                switch(self.runParameters.gamma)
                case 167; gs = '5/3';
                case 140; gs = '7/5';
                case 129; gs = '9/7';
                otherwise; gs = sprintf('%g', self.runParameters.gamma / 100);
                end
                self.infoText.String{1} = sprintf('M=%.3g, theta=%.2g, gamma=%s', self.runParameters.m, self.runParameters.theta, gs);
            end
            if ~isempty(self.F);             self.infoText.String{2} = sprintf('Resolution: %iX x %iT', size(self.F.mass,1), size(self.F.mass,2)); end
            if ~isempty(self.xNormalization); self.infoText.String{3} = sprintf('Xshock = %f', self.xNormalization); end
            if self.pHaveAutovars; self.infoText.String{4} = 'Have autovars'; else; self.infoText.String{4} = 'Lack autovars'; end
            if ~isempty(self.fOscillate);    self.infoText.String{5} = sprintf('Fosc = %f', self.fOscillate); end
            if ~isempty(self.fftPoints);     self.infoText.String{6} = sprintf('FFT frames = [%i %i]', self.fftPoints(1), self.fftPoints(2)); end
            if ~isempty(self.vfallback);     self.infoText.String{7} = sprintf('Vfallback = %f', self.vfallback); end
            if ~isempty(self.fftFundamentalFreq); self.infoText.String{8} = sprintf('Freq. resolution = %f', self.fftFundamentalFreq); end
%%=
%0 ms=%.2g, \theta=%.2g, \gamma=%s
%1 Resolution: %ix x %it
%2 [have / dont have] autovars
%3 Foscillate = %.5g
%4 FFT range: [%i %i]
%5 Fallback velocity: %g
%6 Freq. resolution: %g
%7 
%8 
%9 
%a 
%b 
%c 
%d 
%e 
%f 
%---% 
        end
        
        function executeCurrentAnalysisPhase(self)
            switch self.pCurrentAnalysisPhase
                case 1 % load
                    self.loadRunFrame();
                case 2 % basics
                    self.updateBasics();
                case 3
                    self.pickOscillationPeriod();
                case 4
                    self.pickFFTRange();
                case 5
                    self.runFFT();
                case 6
                    self.insertData();
            end
        end
        
        function loadRunFrame(self)
            % Loads 4D_XYZT from the pwd, checks for glitches or bad restarts and squeeze()es F.

            self.dumpContents();
            self.updateInfoPanel();
            
            if exist('4D_XYZT.mat','file')
                load('4D_XYZT','F');
                % Fix this BS if we encounter it
                if ~isa(F, 'DataFrame'); F = DataFrame(F); save('4D_XYZT','F','-v7.3'); end %#ok<PROP>
                
                self.F = F; %#ok<PROP>
            else
                % ... ? 
            end
            
            if numel(size(self.F.mass)) ~= 4 % it was mistakenly saved while squeezed
                self.print('Glitch: dataframe was saved while squeezed. Automatically fixed.');
                self.F.reshape([size(self.F.mass,1) 1 1 size(self.F.mass,2)]);
                resave = 1;
            else
                resave = 0;
            end
            
            resave = resave + self.F.checkpinteg();
            
            resave = resave + self.F.checkForBadRestartFrame();
            chopped = self.F.chopOutAnomalousTimestep();
            if chopped; resave = resave + 1; end
            
            if resave
                self.print('1 or more BS glitches repaired: Saving correct data');
                F = self.F; %#ok<NASGU,PROP>
                save('4D_XYZT','F','-v7.3');
            end
            
            self.F.squashme(); % convert XYZT to XT
            
            hold off;
            imagesc(log(self.F.mass));
            title('ln(\rho)');
            hold on;
            
            try
                av = load('autovars.mat','autovars');
                
                self.autovars = double(av.autovars); % wtf? true, but this somehow ended up as a single one
                
                %pts = autovars(1:2);
                %pointspace = autovars(3);
                %nlpoint = autovars(4);
                %endpt = autovars(5);
                if self.disableAutoOscil
                    self.autovars(1:2) = av.autovars(1:2) + (size(F.mass,2) - av.autovars(2) + 100); %#ok<PROP>
                end
                
                self.pHaveAutovars = 1;
                self.fftPoints = self.autovars(4:5);
            catch
                self.autovars = []; % prevent old one from hanging on
                self.pHaveAutovars = 0;
            end
            
            % Open the original initializer and look for the fallbackBoost parameter
            % which is necessary to report the correct boost value if the run is to be
            % cold-restarted, which is normally due to the boost being wrong...
            zz = load('SimInitializer_rank0.mat','IC');
            if isfield(zz.IC.ini, 'fallbackBoost')
                zz = zz.IC.ini.fallbackBoost;
            else
                zz = 0;
            end
            self.fallbackBoost = zz;
            
            self.runParameters = RHD_utils.parseDirectoryName(); % the worst way of learning this lol
            
            self.updateInfoPanel();
        end
        
        function updateBasics(self)
            % Updates the X/T position vectors, normalizations and checks for shock-off-grid
            
            self.xVec = self.F.dGrid{1}*(1:size(self.F.mass,1));
            % = 2 pi xshock / uin -> converts to radians per sec
            
            self.coldPos  = RHD_utils.trackColdBoundary(self.F)';
            hitend = self.coldPos > .9*self.xVec(end);
            
            if any(hitend)
                figure(self.figWindow);
                imagesc(log(self.F.mass));
                
                pt = find(hitend, 1);
                trunc = inputdlg('Shock hit end of grid','Frame to truncate at?', [1 10], {num2str(pt)});
                trunc = str2num(trunc{1}); %#ok<ST2NM>
                trunc = min([trunc, size(self.F.mass,2)]);
                trunc = max([1, trunc]);
                if trunc < numel(self.F.time.time)
                    self.F.truncate([], 1:trunc)
                    self.coldPos = self.coldPos(1:trunc);
                end
                trunc = inputdlg('Save', 'Save truncated dataset to disk?', [1 10], {'1'});
                if str2num(trunc{1})
                    F = self.F; %#ok<PROP>
                    save('4D_XYZT.mat','F','-v7.3');
                    clear F;
                end
            end
            
            self.shockPos = trackFront2(self.F.pressure, self.xVec, .5*(self.F.gamma+1)/(self.F.gamma-1))';

            self.xNormalization = self.coldPos(1) - self.shockPos(1);
            
            self.tNormalization = 2*pi*self.xNormalization / self.F.velX(1,1);
            self.timeVec = self.F.time.time / self.tNormalization;

        end
        
        function pickOscillationPeriod(self)
            hold off;
            dlrdt = diff(self.F.mass, 1, 2) ./ self.F.mass(:,1:(end-1));
            imagesc(dlrdt);
            hold on;
            
            lp = max(self.shockPos((end-500):end));
            up = min(self.coldPos((end-500):end));
            vv = round(sort([lp, up])/self.F.dGrid{1});
            q = dlrdt(vv(1):vv(2), (end-500):end);
            q = max(abs(q(:)));
            self.graphAxes.CLim = [-q/2, q/2];

            % Acquire data points demarking an oscillation period
            % This is no longer needed for mode identification but it is useful for making the
            % spectrograms of pure tones awful purty
            
            if self.pHaveAutovars && (self.autovars(2) > self.autovars(1))
                % use them to autoposition for period selection
                xrng = abs(self.autovars(2)-self.autovars(1));
                nt = size(self.F.mass);
                
                xr = [round(nt(2) - 2.5*xrng) nt(2)];
                
                self.graphAxes.XLim = xr;
                up = min(self.shockPos(xr(1):xr(2))) / self.F.dGrid{1};
                dn = max(self.coldPos(xr(1):xr(2))) / self.F.dGrid{1};
                
                self.graphAxes.YLim = [max([up-100, 1]), min([dn + 100,nt(1)])];
            else
                
            end
            
            doauto = self.pHaveAutovars && self.pTryAutoOscil && (self.autovars(2) > 0.9*size(self.F.mass,2)) && (self.autovars(2) < size(self.F.mass,2));
            
            if doauto; haveauto = 2; else
                title('Click 2 points demarking a round trip');
                haveauto = 0; 
            end
            
            [pt, haveauto] = self.acquireValidOscPoint(haveauto);
            self.oscilPoints(1) = RHD_utils.walkdown(self.shockPos, round(pt(1)), 100);
            plot(self.oscilPoints(1), self.shockPos(self.oscilPoints(1)) / self.F.dGrid{1}, 'rX');
            
            [pt, ~] = self.acquireValidOscPoint(haveauto);
            self.oscilPoints(2) = RHD_utils.walkdown(self.shockPos, round(pt(1)), 100);
            plot(self.oscilPoints(2), self.shockPos(self.oscilPoints(2)) / self.F.dGrid{1}, 'rX');
            
            title('');
            
            % In case they click right then left
            self.oscilPoints = sort(self.oscilPoints);
            
            pointspace = 2;
            interframe1 = RHD_utils.projectParabolicMinimum(self.shockPos, self.oscilPoints(1), 0, pointspace);

            if ~isreal(interframe1)
                self.print('WARNING: the shock bounce period appears to be VERY short. Trying again with tighter spaced points.');
                interframe1 = RHD_utils.projectParabolicMinimum(self.shockPos, self.oscilPoints(1), 0, 1);
                
                if ~isreal(interframe1)
                    self.print('WARNING: Parabolas failed for point 1. Using 0th order approximation.');
                    interframe1 = self.oscilPoints(1);
                else % rerun and request a plot this time
                    interframe1 = RHD_utils.projectParabolicMinimum(self.shockPos, self.oscilPoints(1), 1, 1);
                end
                
                pointspace = 1;
            else % rerun with orders to plot
                interframe1 = RHD_utils.projectParabolicMinimum(self.shockPos, self.oscilPoints(1), 1, pointspace);
            end
            
            interframe2 = RHD_utils.projectParabolicMinimum(self.shockPos, self.oscilPoints(2), 0, pointspace);
            if ~isreal(interframe2)
                if pointspace > 1
                    interframe2 = RHD_utils.projectParabolicMinimum(self.shockPos, self.oscilPoints(2), 0, 1);
                end
                
                if ~isreal(interframe2)
                    self.print('WARNING: Parabolas failed for point 2. Using 0th order approximation.');
                    interframe2 = self.oscilPoints(2);
                else % rerun and request a plot this time
                    interframe2 = RHD_utils.projectParabolicMinimum(self.shockPos, self.oscilPoints(2), 1, 1);
                end
            else % rerun with orders to plot
                interframe2 = RHD_utils.projectParabolicMinimum(self.shockPos, self.oscilPoints(2), 1, pointspace);
            end            

            self.projectedPts = [interframe1 interframe2];
            
            tfunda = interp1(1:size(self.timeVec), self.timeVec, [interframe1 interframe2],'linear');

            plot([interframe1 interframe2], self.shockPos(round([interframe1 interframe2])) / self.F.dGrid{1}, 'kv', 'MarkerSize', 10);
            hold off;

            tfunda0 = self.timeVec(self.oscilPoints);
            ffunda0 = 1/(tfunda0(2) - tfunda0(1));

            ffunda = 1/(tfunda(2)-tfunda(1));

            dfreq = abs(ffunda/ffunda0 - 1);
            self.msgBox.String = sprintf('F_{oscillate} = %.3g', ffunda);
            if dfreq > .03
                warndlg({['0th and 2nd order fundamental frequencies differ by ' num2str(100*dfreq) '%'],'consider restarting run or rerunning OSCIL'}, 'Frequency accuracy');
                figure(self.figWindow);
            end
            
            self.fOscillate = ffunda;
            self.updateInfoPanel();
        end
        
        function [pt, autoremain] = acquireValidOscPoint(self, haveauto)
            autoremain = haveauto;
            
            % If we have two autovars points, try the 1st one 1st
            if autoremain == 2
                pt = self.autovars(1);
                q = RHD_utils.walkdown(self.shockPos, round(pt(1)), 100);
                if q < (numel(self.shockPos)-5)
                    autoremain = 1;
                    return;
                end
            end
            
            % If we have one autovars point, try it
            if autoremain == 1
                pt = self.autovars(2);
                q = RHD_utils.walkdown(self.shockPos, round(pt(1)), 100);
                if q < (numel(self.shockPos)-5)
                    autoremain = 0;
                    return;
                end
            end
            
            ptvalid = 0;
            
            P = pointGetter('image');
            while ptvalid == 0
                pt = P.waitForClick();
                q = RHD_utils.walkdown(self.shockPos, round(pt(1)), 100);
                if q < (numel(self.shockPos)-5)
                    if numel(self.oscilPoints) > 0
                        if q ~= self.oscilPoints(1)
                            ptvalid = 1;
                        else
                            self.print('Invalid point: Same as first point');
                        end
                    else
                        ptvalid = 1;
                    end
                else
                    self.print('Invalid point: walks down too close to end of grid.');
                end
            end
            ax = gca();
            ax.Children(1).ButtonDownFcn = [];
        end
        
        function pickFFTRange(self)
            
            hold off;
            plot(self.coldPos - .8*self.xNormalization,'b-');
            hold on;
            
            xmax = max(self.xVec);
            if max(self.coldPos) > .7*xmax
                plot([0 numel(self.coldPos)], [xmax xmax] - .9*self.xNormalization, 'b-x');
            end
            if min(self.shockPos) < .3*xmax
                plot([0 numel(self.coldPos)], self.xVec(4)*[1 1], 'r-x');
            end
            
            yrng = [min(self.shockPos) max(self.coldPos)];
            [rr, ~] = RHD_utils.computeRelativeLuminosity(self.F, self.runParameters.theta);
            rdyn = max(rr) - min(rr);
            rr = (rr - mean(rr))*.3*diff(yrng) / rdyn + sum(yrng)/2; % center on middle of plot
            plot(rr, 'g-.');
            
            
            if self.pHaveAutovars
                %???1
            end
            
            plot(self.shockPos,'r');
            hold off;
            
            self.print('Click the start & end points of the interval to transform.');
            
            
            P = pointGetter('line');
            pt = P.waitForClick();
            P.clickedPoint = [];
            nlpoint(1) = round(pt(1));
            self.msgBox.String = sprintf('1st point: %i\n', pt(1));
            
            pt = P.waitForClick();
            P.clickedPoint = [];
            nlpoint(2) = round(pt(1));
            nlpoint = sort(nlpoint);
            
            self.print(sprintf('Interval: [%i, %i]', int32(nlpoint(1)), int32(nlpoint(2))));
            
            frmOsc = self.projectedPts(2)-self.projectedPts(1);
            
            cycles = round(nlpoint(2)-nlpoint(1))/frmOsc - 1;
            
            endpt = RHD_utils.walkdown(self.shockPos, round(nlpoint(1) + cycles*frmOsc), round(frmOsc));
            stpt = RHD_utils.walkdown(self.shockPos, nlpoint(1), round(frmOsc));
            
            npt = 1+endpt-stpt;
            
            if npt ~= 2*round(npt/2); stpt = stpt-1; end

            self.fftPoints = [stpt endpt];
            
            self.updateInfoPanel();
        end
        
        function runFFT(self)
            % Removes the constant and linear parts from the shock's position,
            % again to minimize spurious spectral junk in the FFT
            timepts = self.timeVec(self.fftPoints(1):self.fftPoints(2));
            pospts  = self.shockPos(self.fftPoints(1):self.fftPoints(2));
            
            [oscil, vfall] = RHD_utils.extractFallback(pospts, timepts * self.tNormalization);
            self.vfallback = self.fallbackBoost + vfall;
            self.print(sprintf('Vfallback (equil rest frame) = %.6f\n', self.vfallback));

            % take and rescale the FFT
            xfourier = 2*abs(fft(oscil / self.xNormalization))/numel(oscil);
            xi = numel(xfourier(2:round(end/2)));

            rth = self.runParameters.theta;

            % This is the time unit for display of FFT results
            tfunda = (timepts(end) - timepts(1));
            
            freqAxis = (1:xi) / tfunda;
            self.fftFundamentalFreq = 1/tfunda;
            
            % Throw up fft of shock position
            hold off;
            plot(freqAxis, xfourier(2:round(end/2)),'b-');
            hold on

            % Compute the luminosity on the given interval and normalize it by L(t=0) and fft it
            [rr, self.radNormalization] = RHD_utils.computeRelativeLuminosity(self.F, rth);
            rft = 2*fft(rr(self.fftPoints(1):self.fftPoints(2)))' / numel(oscil);

            plot(freqAxis,abs(rft(2:round(end/2))), 'r-');

            %lpp = max(rr(stpt:endpt))-min(rr(stpt:endpt));
 
            grid on;
            
            xlabel('Normalized frequency');
            ylabel('Normalized amplitudes');
            title('Spectral analysis results');
            
            self.datablock = self.tagModes(xfourier, rft);
            self.datablock = self.sanityCheckModes(self.datablock);
            
            self.updateInfoPanel();
        end
        
        function insertData(self)
            
            % write autovars here
            autovars = double([round(self.projectedPts), 2, self.fftPoints]); %#ok<NASGU,PROP>
            save('autovars.mat','autovars');
            
            % get convergence level estimate from user
            if self.pAutomaticMode
                conq = 5;
            else
                conq = inputdlg('Convergence quality (1-5)?','figure selection', [1 10], {'0'});
                conq = str2double(conq);
            end
            
            if isempty(conq) || isnan(conq) % user clicked 'cancel' or entered garbage
                return;
            end
            
            numprops = [conq, self.vfallback, round(self.xNormalization / self.F.dGrid{1}), self.fftFundamentalFreq];
            
            % These have been normalized by the analyzers, using comparatively crude numeric
            % normalizations computed from frame 0; The FMHandler assumes they are unnormalized
            % and applies its own precision-calculated normalizations
            self.datablock(:,2) = self.datablock(:,2) * self.xNormalization;
            
            % don't do this until we're ready to do a complete re-analysis because all currently
            % stored values are numerically normalized.
            %self.datablock(:,3) = self.datablock(:,3) * self.radNormalization;
                        
            % Go through this rigamarole to access the next frame up and insert the data myself
            proto = 'f%i.insertPointNew(%f, %f, %s, %s);';
            if self.runParameters.gamma == 167
                if evalin('base', 'exist(''f53'', ''var'')')
                    p = sprintf(proto, 53, self.runParameters.m, self.runParameters.theta, mat2str(self.datablock), mat2str(numprops));
                    evalin('base', p);
                    %f53.insertPointNew(self.runParameters.m, self.runParameters.theta, self.datablock, conq);
                elseif exist('self', 'var')
                    
                else
                    disp('Access to FMHandler directly is required to insert data.\n');
                end
            elseif self.runParameters.gamma == 140
                if evalin('base', 'exist(''f75'', ''var'')')
                    p = sprintf(proto, 75, self.runParameters.m, self.runParameters.theta, mat2str(self.datablock), mat2str(numprops));
                    evalin('base', p);
                    %f75.insertPointNew(runparams.m, runparams.theta, self.datablock, conq);
                else
                    disp('Access to FMHandler directly is required to insert data.\n');
                end
            elseif self.runParameters.gamma == 129
                if evalin('base', 'exist(''f97'', ''var'')')
                    p = sprintf(proto, 97, self.runParameters.m, self.runParameters.theta, mat2str(self.datablock), mat2str(numprops));
                    evalin('base', p);
                    %f97.insertPointNew(runparams.m, runparams.theta, self.datablock, conq);
                else
                    disp('Access to FMHandler directly is required to insert data.\n');
                end
            else
                disp('Strange, an adiabatic index that is not 5/3, 7/5 or 9/7?\n');
            end

            od = pwd();
            cd ..;
            if (conq == 5) && (self.automaticMode == 0)
                xd = pwd();
                if strcmp(xd((end-8):end), 'radshocks') == 0
                    eval(sprintf('!mv %s ../radshocks', od));
                end
                self.print('Automatically moved conv lvl = 5 run to ../radshocks');
            end
            
        end
        
        function datablock = tagModes(self, xfourier, Lfourier)
            
            fatdot = [];
            raddot = [];
            
            % helpers to identify gaussian peaks
            ispeak = @(y, xi) (y(xi) > y(xi+1)) & (y(xi) > y(xi-1)) & ( y(xi) > 2*sum(y(xi+[-2, -1, 1, 2])) );
            isgausspeak = @(mag, center, std) (abs(center) < .55) & (std < 3);
            
            xresid = xfourier(2:round(end/2));
            % wipe out very low freq noise potentials
            xresid(1:4) = xresid(5);
            
            for n = 1:12
                % Pick the highest peak
                [~, pkidx] = max(xresid);
                p = pkidx + 1;
                
                if p < 5; continue; end
                if p + 4 > numel(xresid); break; end
                
                [mag, center, std] = RHD_utils.gaussianPeakFit(xfourier, p);
                
                if isgausspeak(mag, center - p, std)
                    fatdot(end+1,:) = [center-1, mag, std];
                    
                    % extract the radiance spectral peak as well
                    [mag, center, std] = RHD_utils.gaussianPeakFit(abs(Lfourier), p);
                    raddot(end+1,:) = [center-1, mag, std];
                end
                
                % Truncate the entire peak
                xresid = RHD_utils.chopPeakForSearch(xresid, pkidx);
            end
            
%            spec_residual = RHD_utils.subtractKnownPeaks(xfourier(2:end/2), fatdot);
            
            % convert the indexes in fatdot to frequencies
            fatdot(:,[1 3]) = fatdot(:,[1 3]) * self.fftFundamentalFreq;
            raddot(:,[1 3]) = raddot(:,[1 3]) * self.fftFundamentalFreq;
            
            % Drop the detected peaks onto the graph and print about them
            plot(fatdot(:,1), fatdot(:,2), 'kv', 'MarkerSize', 8);
            xlim([0 min(10, 2*max(fatdot(:,1)))]);
            
            runpar = self.runParameters;
            
            % Plot frequency bin markers
            wpk = [.87 2.85 5 7 9 11 13 15];
            switch self.runParameters.gamma
                case 167; wpk = wpk * (1 + 1.14 / runpar.m + 1.45/runpar.m^2) * (1 - .020*runpar.theta) * .261;
                case 140; wpk = wpk * (1 + 2.50 / runpar.m) * (1 - .042*runpar.theta) * .186;
                case 129; wpk = wpk * (1 + 2.84 / runpar.m) * (1 - .060*runpar.theta) * .150;
            end
            lm = max(abs(Lfourier(2:(end/2))));
            wpk = [wpk; wpk];
            lm = lm*[0; 1]*[1 1 1 1 1 1 1 1];
            plot(wpk, lm, 'g-');
            
            
            self.print(sprintf("Frequency resolution = +-%f\n", self.fftFundamentalFreq));
            
            SA = SpectralAnalyzer(fatdot, raddot);
            SA.aprioriTag(@(w) RHD_utils.assignModeNumber(w, runpar.m, runpar.gamma, runpar.theta));
            SA.thereCanBeOnlyOne();
            SA.testFakeIntermodulation();
            SA.tagIntermodulation;
            
            SA.printTable();
            
            legend('X_{shock} spectrum',  'Relative luminosity spectrum', 'Detected peaks');
            
            drawnow;
            
            hold off;
            pwd

            datablock = zeros([11 3]); % rows of [frequency xshock_amp radiance_amp] for mode F to 10
            
            for q = 1:numel(SA.table)
                tau = SA.table(q);
                if (tau.m >= 0) && (tau.m < 11)
                    if tau.hd == 1
                        datablock(tau.m+1, :) = [tau.f/self.tNormalization, tau.dx, tau.dl];
                    end
                end
                
            end

            self.updateInfoPanel();
        end
        
        function y = sanityCheckModes(self, blk)
            % Look for absurd values fitted by the Gaussian fitter
            
            % Absurd amplitudes
            a = find(blk(:,2) > 2*self.xNormalization);
            % Unreasonable luminance fluctuation (these are bounded by roughly 1)
            b = find(blk(:,3) > 5);
            
            keeps = ones([size(blk,1) 1]);
            
            if any(a)
                fprintf('Unreasonable shock position amplitude fit found: dumping\n');
                for j = 1:numel(a)
                    fprintf('f=%f, drho=%f, dluminance=%f', blk(j,1), blk(j,2), blk(j,3));
                end
                keeps(a) = 0;
            end
            if any(b)
                fprintf('Unreasonable luminance amplitudes found: zeroing\n');
                blk(b,3) = 0;
            end

            y = blk(find(keeps),:);
        end
        
        function runFullAnalysis(self)
            % Iterates through all the run phases in sequence
            if self.pCurrentAnalysisPhase ~= 0; return; end
            
            self.runAnalysisPhase = 1; % load
            
            self.runAnalysisPhase = 2; % basics
            
            self.pTryAutoOscil = 1;
            self.runAnalysisPhase = 3; % run oscil (it skips input itself if possible)
            self.pTryAutoOscil = 0;
                        
            if ( bitand(self.pAutomaticMode,1) == 0) || (self.pHaveAutovars == 0)
                self.runAnalysisPhase = 4; % pick fft range
            else % this is curious, interrupt for manual input
                if (self.fftPoints(2) < .95*size(self.F.mass,2)) || any(self.fftPoints > size(self.F.mass,2))
                    self.runAnalysisPhase = 4;
                end
            end 
            
            self.runAnalysisPhase = 5; % fft 
            
            if bitand(self.pAutomaticMode,2)
                self.runAnalysisPhase = 6; % save     
            end
        end
        
        
    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                      P R O T E C T E D    [M]
        
        function print(self, s)
            disp(s);
            self.msgBox.String = s;
        end

        function cbRunButton(self, src, data)
            self.runFullAnalysis();
        end
        
        function cbDynrngPlus(self, src, data)
            self.graphAxes.CLim = self.graphAxes.CLim * 5;
        end
            
        function cbDynrngMinus(self, src, data)
            self.graphAxes.CLim = self.graphAxes.CLim / 5;
        end
        
        function cbSequenceBtns(self, src, data)
            if self.pCurrentAnalysisPhase ~= 0; return; end
            if strcmp(src.Tag, 'loadbutton'); self.runAnalysisPhase = 1; end
            if strcmp(src.Tag, 'basicsbutton'); self.runAnalysisPhase = 2; end
            if strcmp(src.Tag, 'oscilbutton'); self.runAnalysisPhase = 3; end
            if strcmp(src.Tag, 'rangebutton'); self.runAnalysisPhase = 4; end
            if strcmp(src.Tag, 'fftbutton'); self.runAnalysisPhase = 5; end
            if strcmp(src.Tag, 'writebutton'); self.runAnalysisPhase = 6; end
        end
        
        function cbShowStatus(self, src, data)
            if data.Source.Value == 1 % pushed
                self.toolsPanel.Visible = 'on';
            else
                self.toolsPanel.Visible = 'off';
            end
        end

        function queryPlotCallback(self, src, data)
            
            if strcmp(src.Tag, 'plotfreq'); evalin('base',sprintf('f%s.emitDominantFreqPlot(1, 0, 2)',self.pltGamma)); end
            if strcmp(src.Tag, 'plotdx'); evalin('base',sprintf('f%s.emitDominantFreqPlot(2, 0, 2)',self.pltGamma)); end
            if strcmp(src.Tag, 'plotlum'); evalin('base',sprintf('f%s.emitDominantFreqPlot(3, 0, 2)',self.pltGamma)); end
            
            if strcmp(src.Tag, 'gam53'); self.pltGamma = '53'; end
            if strcmp(src.Tag, 'gam75'); self.pltGamma = '75'; end
            if strcmp(src.Tag, 'gam97'); self.pltGamma = '97'; end
            
            if strcmp(src.Tag, 'ptquery')
                if src.Value == 1
                    self.graphAxes.Children(2).ButtonDownFcn = @self.pointQueries;
                else
                    self.graphAxes.Children(2).ButtonDownFcn = '';
                end
            end 
            if strcmp(src.Tag, 'plistgen')
                if src.Value == 1
                    self.graphAxes.Children(2).ButtonDownFcn = @self.pointQueries;
                else
                    self.graphAxes.Children(2).ButtonDownFcn = '';
                end
            end 
        end
        
        function pointQueries(self, src, data)
            ip = data.IntersectionPoint;
            
            ip(2) = .25 * round(ip(2)/.25);
            ip(1) = .05 * round(ip(1)/.05);
            % point query
            if self.queryButtons{7}.Value; evalin('base', sprintf('f%s.searchMode(%f, %f)', self.pltGamma, ip(2), ip(1))); end
            if self.queryButtons{8}.Value % plist generator
                p = evalin('base',sprintf('f%s.findPoint(%f,%f)', self.pltGamma, ip(2), ip(1)));
                if strcmp(self.pltGamma,'53'); thermt = 1; end
                if strcmp(self.pltGamma,'75'); thermt = 2; end
                if strcmp(self.pltGamma,'97'); thermt = 3; end 
                fprintf('%.4g, %.4g, %f, %i; ', ip(2), ip(1), evalin('base',sprintf('f%s.fallbackRate(%i)',self.pltGamma,int32(p))),thermt);
            end
        end
        
        function testCallback(self, src, data)
            disp('Test callback hit:');
            disp(src);
            src.get
        end
        
        function handleAnalyzerResize(self, src, data)
            rez = src.Position;

            pix = 1 ./ self.figWindow.OuterPosition(3:4);
            
            dy = self.ctrlPanel.Position(4) / rez(4); 
            
            self.graphAxes.Position = [.10 (dy + 48*pix(2) + 1*self.pCharH/rez(4)) .85 (.95-dy - 64*pix(2))];
            
        end
        
        function interrogateSystemDPI(self)
            % Grabs the groot graphics handle, reads .ScreenPixelsPerInch & sets up some private
            % constants used to scaling text on screen
            core = groot;
            DPI = core.ScreenPixelsPerInch;
            txtH = round(.14*DPI); % assuming 10pt font
            charW = round(.14*.6*DPI); % guess avg/char width
            
            self.pDPI = DPI;
            self.pCharW = charW;
            self.pCharH = txtH;
            self.pLineH = round(.14*1.2*DPI);
        end
    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %                                                 S T A T I C    [M]
        
    end%PROTECTED
    
end%CLASS
