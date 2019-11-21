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
        
        F; % the data (F)rame, with a nice short variable name :)
        
        runParameters;
        
        xNormalization, xVec;
        tNormalization, timeVec;
        shockPos; coldPos;
        fallbackBoost, vfallback;
        
        autovars;
        
        oscilPoints;
        projectedPts;
        
        fOscillate;
        
        fftPoints;
        fftFundamentalFreq;
        
        datablock;
    end %PUBLIC
    
    properties(Dependent = true)
        runAnalysisPhase;
    end
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
        pDPI;
        pCharW, pCharH, pLineH;
        
        pCurrentAnalysisPhase; % 0 = idle, 1 = load, 2= basics, 3 = oscil period, 4 = fft, 5 = commit
        pDisableSequenceButtons;
        pHaveAutovars;
        
        pAbortSequence;
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
            drawnow();
            
            self.executeCurrentAnalysisPhase();
            
            for j = 1:numel(bvals); self.ctrlButtons{j}.Value = 0; end
            drawnow();
            
            self.pCurrentAnalysisPhase = 0;
        end
        
        function p = get.runAnalysisPhase(self)
            p = self.pCurrentAnalysisPhase;
        end
    
    end%GET/SET
    
    %===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
        
        function dbgclear(self)
            self.pCurrentAnalysisPhase = 0;
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
            
            self.setupControlPanel();
            
            self.pCurrentAnalysisPhase = 0;
            self.pAbortSequence = 0;
            % Buttons: [run full sequence] [auto = 2] [pick oscil period] [pick fft range] [fft] [save] 
            
            self.graphAxes = gca();
            
            spix = self.figWindow.OuterPosition;
            pix = 1 ./ self.figWindow.OuterPosition(3:4);
            
            dy = self.ctrlPanel.Position(4) / spix(4); 
            
            self.graphAxes.Position = [.1 (dy + 48*pix(2)) .8 (.95-dy - 64*pix(2))];
        end
        
        function setupControlPanel(self)
            dx = self.pCharW;
            
            self.ctrlPanel = uipanel(self.figWindow,'Title','Analysis tools', 'units', 'pixels', 'position', [8, 8, 60*self.pCharW, 4*self.pLineH]);
            
            self.ctrlButtons{1} = uicontrol(self.ctrlPanel, 'Style','togglebutton','String','RUN','tag','runbutton','position',[8,      8, 6*dx, 1.5*self.pCharH], 'callback', @self.cbRunButton);
            
            self.ctrlButtons{2} = uicontrol(self.ctrlPanel, 'Style','togglebutton','String','LOAD','tag','loadbutton','position',[8+7*dx, 8, 6*dx, 1.5*self.pCharH], 'callback', @self.cbLoadButton);
            self.ctrlButtons{3} = uicontrol(self.ctrlPanel, 'Style','togglebutton','String','BASICS','tag','basicsbutton','position',[8+14*dx, 8, 6*dx, 1.5*self.pCharH], 'callback', @self.cbBasicButton);
            self.ctrlButtons{4} = uicontrol(self.ctrlPanel, 'Style','togglebutton','String','OSCIL','tag','oscilbutton','position',[8+21*dx, 8, 6*dx, 1.5*self.pCharH], 'callback', @self.cbOscilButton);
            self.ctrlButtons{5} = uicontrol(self.ctrlPanel, 'Style','togglebutton','String','RANGE','tag','rangebutton','position',[8+28*dx, 8, 6*dx, 1.5*self.pCharH], 'callback', @self.cbXformButton);
            self.ctrlButtons{6} = uicontrol(self.ctrlPanel, 'Style','togglebutton','String','FFT','tag','fftbutton','position',[8+35*dx, 8, 6*dx, 1.5*self.pCharH], 'callback', @self.cbFFTButton);
            self.ctrlButtons{7} = uicontrol(self.ctrlPanel, 'Style','togglebutton','String','WRITE','tag','fftbutton','position',[8+42*dx, 8, 6*dx, 1.5*self.pCharH], 'callback', @self.cbWriteButton);
            self.msgBox         = uicontrol(self.ctrlPanel, 'Style','text','String','info', 'position',[0, 8+2*self.pCharH, 49*dx, 1.2*self.pCharH]);
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

            if exist('4D_XYZT.mat','file')
                load('4D_XYZT','F');
                % Fix this BS if we encounter it
                if ~isa(F, 'DataFrame'); F = DataFrame(F); save('4D_XYZT','F'); end
                
                self.F = F; %#ok<CPROP>
            else
                % ... ? 
            end
            
            br = self.F.checkForBadRestartFrame();
            chopped = self.F.chopOutAnomalousTimestep();
            
            if chopped
                disp('User indicates acceptable truncation of bad restart: Saving correct data.');
                F = self.F;
                save('4D_XYZT','F');
            end
            
            self.F.squashme(); % convert XYZT to XT
            
            hold off;
            imagesc(log(self.F.mass));
            title('ln(\rho)');
            hold on;
            
            try
                load('autovars.mat');
                
                self.autovars = double(autovars); %#ok<CPROP> % wtf? true, but this somehow ended up as a single one
                
                %pts = autovars(1:2);
                %pointspace = autovars(3);
                %nlpoint = autovars(4);
                %endpt = autovars(5);
                
                self.pHaveAutovars = 1;
            catch crap
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
            
            q = dlrdt(1:end, (end-500):end);
            q = max(abs(q(:)));
            self.graphAxes.CLim = [-q/2, q/2];

            % Acquire data points demarking an oscillation period
            % This is no longer needed for mode identification but it is useful for making the
            % spectrograms of pure tones awful purty
            
            if self.pHaveAutovars
                % use them to autoposition for period selection
                xrng = self.autovars(2)-self.autovars(1);
                nt = size(self.F.mass);
                
                xr = [round(nt(2) - 1.5*xrng) nt(2)];
                
                self.graphAxes.XLim = xr;
                up = min(self.shockPos(xr(1):xr(2))) / self.F.dGrid{1};
                dn = max(self.coldPos(xr(1):xr(2))) / self.F.dGrid{1};
                
                self.graphAxes.YLim = [max([up-100, 1]), min([dn + 100,nt(1)])];
            else
                
            end
            
            title('Click 2 points demarking a round trip')
            
            P = pointGetter();
            pt = P.waitForClick();
            P.clickedPoint = [];
            pts(1) = RHD_utils.walkdown(self.shockPos, round(pt(1)), 100);
            
            plot(pts(1), self.shockPos(pts(1)) / self.F.dGrid{1}, 'rX');
            
            pt = P.waitForClick();
            P.clickedPoint = [];
            pts(2) = RHD_utils.walkdown(self.shockPos, round(pt(1)), 100);
            
            % In case they click right then left
            self.oscilPoints = sort(pts);

            plot(pts(2), self.shockPos(pts(2)) / self.F.dGrid{1}, 'rX');
            
            pointspace = 2;
            interframe1 = RHD_utils.projectParabolicMinimum(self.shockPos, pts(1), 0, pointspace);

            if ~isreal(interframe1)
                disp('WARNING: the shock bounce period appears to be VERY short. Trying again with tighter spaced points.');
                interframe1 = RHD_utils.projectParabolicMinimum(self.shockPos, pts(1), 0, 1);
                
                if ~isreal(interframe1)
                    disp('WARNING: Parabolas failed for point 1. Using 0th order approximation.');
                    interframe1 = pts(1);
                else % rerun and request a plot this time
                    interframe1 = RHD_utils.projectParabolicMinimum(self.shockPos, pts(1), 1, 1);
                end
                
                pointspace = 1;
            else % rerun with orders to plot
                interframe1 = RHD_utils.projectParabolicMinimum(self.shockPos, pts(1), 1, pointspace);
            end
            
            interframe2 = RHD_utils.projectParabolicMinimum(self.shockPos, pts(2), 0, pointspace);
            if ~isreal(interframe2)
                if pointspace > 1
                    interframe2 = RHD_utils.projectParabolicMinimum(self.shockPos, pts(2), 0, 1);
                end
                
                if ~isreal(interframe2)
                    disp('WARNING: Parabolas failed for point 2. Using 0th order approximation.');
                    interframe2 = pts(2);
                else % rerun and request a plot this time
                    interframe2 = RHD_utils.projectParabolicMinimum(self.shockPos, pts(2), 1, 1);
                end
            else % rerun with orders to plot
                interframe2 = RHD_utils.projectParabolicMinimum(self.shockPos, pts(2), 1, pointspace);
            end            

            self.projectedPts = [interframe1 interframe2];
            
            tfunda = interp1(1:size(self.timeVec), self.timeVec, [interframe1 interframe2],'linear');

            plot([interframe1 interframe2], self.shockPos(round([interframe1 interframe2])), 'kv', 'MarkerSize', 10);
            hold off;

            tfunda0 = self.timeVec(pts);
            ffunda0 = 1/(tfunda0(2) - tfunda0(1));

            ffunda = 1/(tfunda(2)-tfunda(1));

            dfreq = abs(ffunda/ffunda0 - 1);
            self.msgBox.String = sprintf('F_{oscillate} = %.3g', ffunda);
            if dfreq > .03
                warndlg({['0th and 2nd order fundamental frequencies differ by ' num2str(100*dfreq) '%'],'consider restarting run or rerunning OSCIL'}, 'Frequency accuracy');
            end
            
            
            self.fOscillate = ffunda;
        end
        
        function pickFFTRange(self)
            
            hold off;
            plot(self.coldPos - .8*self.xNormalization,'b-');
            hold on;
            
            xmax = max(self.xVec);
            if max(self.coldPos) > .9*xmax
                plot([0 numel(self.coldPos)], [xmax xmax] - .9*self.xNormalization, 'b-x');
            end
            if min(self.shockPos) < .05*xmax
                plot([0 numel(self.coldPos)], self.xVec(4)*[1 1], 'r-x');
            end
            
            if self.pHaveAutovars
                %???1
            end
            
            plot(self.shockPos,'r');
            hold off;
            
            
            disp('Click the start & end points of the interval to transform.');
            
            P = pointGetter();
            pt = P.waitForClick();
            P.clickedPoint = [];
            nlpoint(1) = round(pt(1));
            self.msgBox.String = sprintf('1st point: %i\n', pt(1));
            
            pt = P.waitForClick();
            P.clickedPoint = [];
            nlpoint(2) = round(pt(1));
            nlpoint = sort(nlpoint);
            
            self.msgBox.String = sprintf('Interval: [%i, %i]', int32(nlpoint(1)), int32(nlpoint(2)));
            
            frmOsc = self.projectedPts(2)-self.projectedPts(1);
            
            cycles = round(nlpoint(2)-nlpoint(1))/frmOsc - 1;
            
            endpt = RHD_utils.walkdown(self.shockPos, round(nlpoint(1) + cycles*frmOsc), round(frmOsc));
            stpt = RHD_utils.walkdown(self.shockPos, nlpoint(1), round(frmOsc));
            
            npt = 1+endpt-stpt;
            
            if npt/2 ~= round(npt/2); stpt = stpt-1; end
            
            self.fftPoints = [stpt endpt];
        end
        
        function runFFT(self)
            % Removes the constant and linear parts from the shock's position,
            % again to minimize spurious spectral junk in the FFT
            timepts = self.timeVec(self.fftPoints(1):self.fftPoints(2));
            pospts  = self.shockPos(self.fftPoints(1):self.fftPoints(2));
            
            [oscil, vfall] = RHD_utils.extractFallback(pospts, timepts * self.tNormalization);
            self.vfallback = self.fallbackBoost + vfall;
            self.msgBox.String = sprintf('Vfallback (equil rest frame) = %.6f\n', self.vfallback);

            % take and rescale the FFT
            xfourier = 2*abs(fft(oscil))/numel(oscil);
            xi = numel(xfourier(2:end/2));

            rth = self.runParameters.theta;

            % This is the time unit for display of FFT results
            tfunda = (timepts(end) - timepts(1));
            
            freqAxis = (1:xi) / tfunda;
            self.fftFundamentalFreq = 1/tfunda;
            
            % Throw up fft of shock position
            hold off;
            plot(freqAxis, xfourier(2:end/2)/self.xNormalization,'b-');
            hold on

            % Compute the luminosity on the given interval and normalize it by L(t=0) and fft it
            rr = RHD_utils.computeRelativeLuminosity(self.F, rth);
            rft = 2*fft(rr(self.fftPoints(1):self.fftPoints(2)))' / numel(oscil);

            plot(freqAxis,abs(rft(2:end/2)), 'r-');

            %lpp = max(rr(stpt:endpt))-min(rr(stpt:endpt));
 
            grid on;
            
            self.datablock = self.tagModes(xfourier, rft);
            self.sanityCheckModes(self.datablock);
            
            
        end
        
        function insertData(self)
            
            % write autovars here
            autovars = double([round(self.projectedPts), 2, self.fftPoints]); %#ok<PROP>
            save('autovars.mat','autovars');
            
            % get convergence level estimate from user
            conq = inputdlg('Convergence quality (1-5)?','figure selection', [1 10], {'0'});
            conq = str2double(conq);
            
            % Go through this rigamarole to access the next frame up and insert the data myself
            proto = 'f%i.insertPointNew(%f, %f, %s, %i);';
            if self.runParameters.gamma == 167
                if evalin('base', 'exist(''f53'', ''var'')')
                    p = sprintf(proto, 53, self.runParameters.m, self.runParameters.theta, mat2str(self.datablock), conq);
                    evalin('base', p);
                    %f53.insertPointNew(self.runParameters.m, self.runParameters.theta, self.datablock, conq);
                elseif exist('self', 'var')
                    
                else
                    disp('Access to FMHandler directly is required to insert data.\n');
                end
            elseif self.runParameters.gamma == 140
                if evalin('base', 'exist(''f75'', ''var'')')
                    p = sprintf(proto, 75, self.runParameters.m, self.runParameters.theta, mat2str(self.datablock), conq);
                    evalin('base', p);
                    %f75.insertPointNew(runparams.m, runparams.theta, self.datablock, conq);
                else
                    disp('Access to FMHandler directly is required to insert data.\n');
                end
            elseif self.runParameters.gamma == 129
                if evalin('base', 'exist(''f97'', ''var'')')
                    p = sprintf(proto, 97, self.runParameters.m, self.runParameters.theta, mat2str(self.datablock), conq);
                    evalin('base', p);
                    %f97.insertPointNew(runparams.m, runparams.theta, self.datablock, conq);
                else
                    disp('Access to FMHandler directly is required to insert data.\n');
                end
            else
                disp('Strange, an adiabatic index that is not 5/3, 7/5 or 9/7?\n');
            end

            
        end
        
        function datablock = tagModes(self, xfourier, Lfourier)
            
            fatdot = [];
            raddot = [];
            possibleFmode = 0;
            
            % helpers to identify gaussian peaks
            ispeak = @(y, xi) (y(xi) > y(xi+1)) & (y(xi) > y(xi-1)) & ( y(xi) > 2*sum(y(xi+[-2, -1, 1, 2])) );
            isgausspeak = @(mag, center, std) (abs(center) < .55) & (std < 3);
            
            xresid = xfourier(2:end/2);
            for n = 1:10
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
            
            spec_residual = RHD_utils.subtractKnownPeaks(xfourier(2:end/2), fatdot);
            
            % convert the indexes in fatdot to frequencies
            fatdot(:,1) = fatdot(:,1) * self.fftFundamentalFreq;
            raddot(:,1) = raddot(:,1) * self.fftFundamentalFreq;
            
            % Drop the detected peaks onto the graph and print about them
            plot(fatdot(:,1), fatdot(:,2)/self.xNormalization, 'kv', 'MarkerSize', 8);
            xlim([0 2*max(fatdot(:,1))]);
            
            fprintf("Frequency resolution = +-%f\n", self.fftFundamentalFreq);
            modenames = cell([n 1]);
            
            % First round of mode tagging
            for n = 1:size(fatdot,1)
                fi = fatdot(n,1);
                
                if ~isempty(modenames{n}); continue; end
                
                s0 = RHD_utils.assignModeName(fi, self.runParameters.m, self.runParameters.gamma, self.runParameters.theta);
                
                if numel(s0) > 0
                    modenames{n} = s0;
                    % Attempt to identify harmonic distortion
                    for u=1:size(fatdot,1)
                        if isempty(modenames{u})
                            for hd = 2:9
                                if abs(fatdot(u,1) - hd*fatdot(n,1)) < 1.5*self.fftFundamentalFreq
                                    modenames{u} = sprintf('%sx%i', modenames{n}, int32(hd));
                                end
                            end
                        end
                    end
                    
                end
            end
            
            pklist = [];
            for n = 1:size(fatdot, 1); if ~isempty(modenames{n}); if numel(modenames{n}) <= 2; pklist(end+1) = n; end; end; end
            nPeaks = numel(pklist);
            
            % Identify 2nd order intermodulation
            for n = 1:size(fatdot, 1)
                if isempty(modenames{n}) % not identified as a mode
                    for u=1:nPeaks
                        for v = (u+1):nPeaks
                            if abs(fatdot(pklist(u),1) + fatdot(pklist(v),1) - fatdot(n,1)) < self.fftFundamentalFreq
                                modenames{n} = [modenames{pklist(u)} '+' modenames{pklist(v)}];
                            end
                            
                            if abs(fatdot(pklist(u),1) - fatdot(pklist(v),1) - fatdot(n,1)) < self.fftFundamentalFreq
                                modenames{n} = [modenames{pklist(u)} '-' modenames{pklist(v)}];
                            end
                            
                            if abs(fatdot(pklist(v),1) - fatdot(pklist(u),1) - fatdot(n,1)) < self.fftFundamentalFreq
                                modenames{n} = [modenames{pklist(u)} '-' modenames{pklist(v)}];
                            end
                        end
                    end
                end
                
            end
            
            for n = 1:size(fatdot,1)
                s0 = modenames{n};
                fi = fatdot(n,1);
                fprintf('Spectral peak at f = %f = %f f0 has magnitude %f', fi, fi / self.fOscillate, fatdot(n,2));
                if numel(s0) > 0
                    if numel(s0) > 2
                        if s0(3) == 'x'
                            fprintf(' - %s harmonic distortion\n', s0);
                        else
                            fprintf(' - %s intermodulation\n', s0);
                        end
                    else
                        fprintf(' - %s peak identified\n', s0);
                    end
                else
                    fprintf('\n');
                end
            end
            
            legend('X_{shock} spectrum',  'Relative luminosity spectrum', 'Detected peaks');
            
            drawnow;
            
            hold off;
            pwd
            
            modelist = {'F','1O','2O','3O','4O','5O','6O','7O','8O','9O','10O'};
            datablock = zeros([11 3]); % rows of [frequency xshock_amp radiance_amp] for mode F to 10
            
            for q = 1:numel(pklist)
                qp = pklist(q);
                for k=1:numel(modelist)
                    if strcmp(modenames{qp},modelist{k}) == 1
                        datablock(k, :) = [fatdot(qp,1)/self.tNormalization, fatdot(qp,2), raddot(qp,2)];
                    end
                end
                
            end

        end
        
        function y = sanityCheckModes(self, blk)
            % Look for absurd values fitted by the Gaussian fitter
            a = find(blk(:,2) > 10);
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
        
    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                      P R O T E C T E D    [M]

        function cbRunButton(self, src, data)
            % Iterates through all the run phases in sequence
            if self.pCurrentAnalysisPhase ~= 0; return; end
            
            self.runAnalysisPhase = 1; % load
            
            self.runAnalysisPhase = 2; % basics
            
            if self.pHaveAutovars && (self.autovars(2) / size(self.F.mass,2) > .95)
                % no need to run this if the oscillation interval is chosen from end of run anyway
            else
                self.runAnalysisPhase = 3; % oscil
            end
            
            self.runAnalysisPhase = 4; % fft rng
            
            self.runAnalysisPhase = 5; % fft 
            
            %self.runAnalysisPhase = 6; % save     
        end
        
        function cbLoadButton(self, src, data)
            if self.pCurrentAnalysisPhase ~= 0; return; end
            self.runAnalysisPhase = 1;
        end
        
        function cbBasicButton(self, src, data)
            if self.pCurrentAnalysisPhase ~= 0; return; end
            self.runAnalysisPhase = 2;
        end
        
        function cbOscilButton(self, src, data)
            if self.pCurrentAnalysisPhase ~= 0; return; end
            self.runAnalysisPhase = 3;
        end
        
        function cbXformButton(self, src, data)
            if self.pCurrentAnalysisPhase ~= 0; return; end
            self.runAnalysisPhase = 4;
        end
        
        function cbFFTButton(self, src, data)
            if self.pCurrentAnalysisPhase ~= 0; return; end
            self.runAnalysisPhase = 5;
        end
        
        function cbWriteButton(self, src, data)
            if self.pCurrentAnalysisPhase ~= 0; return; end
            self.runAnalysisPhase = 6;
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
