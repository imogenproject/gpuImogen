classdef FMHandler2 < handle
    % Class annotation template for creating new classes.
    %___________________________________________________________________________________________________
    
    %===================================================================================================
    properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        dataFilename;
        dataDir;

        MachRange; % [a b c] for Mach outputs a:b:c
        thetaRange;% same
        
        gamma;
        
        dangerous_autoOverwrite;
        
        fallbackRate;
        nShockCells;
        spectralResolution;
        
        convergenceLevel;
    end %PUBLIC
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = public) %                P R O T E C T E D [P]
        machPts; % X
        thetaPts; % Y
        
        fNormMode;
        
        fnormPts;   % Coefficients for frequency normalization
        xnormPts;   % X_shock coefficients for position amplitude normalization
        radnormPts; % Equilibrium radiance for luminance amplitude normalization
        
        peakFreqs;  % Gaussian-estimate peaks in frequency 
        peakMassAmps;%Xshock displacement amplitudes
        peakLumAmps;% Integrated luminosity fluctuation amplitudes
    end %PROTECTED
    
    properties (SetAccess = protected, GetAccess = protected)


    end
    
    %===================================================================================================
    methods %                                                                     G E T / S E T  [M]
    end%GET/SET
    
    %===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
        
        function self = FMHandler2(dfile)
            if nargin==1 
                load(dfile,'self');
                self.dataFilename = dfile;
                self.dataDir = pwd();
            end
            self.dangerous_autoOverwrite = 0;
           
        end
        
        function saveme(self)
            if ~isempty(self.dataFilename)
                save([self.dataDir '/' self.dataFilename], 'self');
            end
        end

        function autoanalyzeEntireDirectory(self, startat, A)
            % autoanalyzeEntireDirectory(self, startat, RHD_Analyzer)
            % Searches the cwd for any directories matching RADHD*gam###
            % where ### is the adiabatic index of myself x 100 and
            % and runs RHD_1danalysis in them with rhdAutodrive=2 (auto
            % everything except for spectral analysis interval)
            % if nargin < 3, creates its own RHD_Analyzer, otherwise uses the passed one.
            if nargin < 2; startat = 1; end
            dlist = dir(sprintf('RADHD*gam%3i',round(100*self.gamma)));
            
            fprintf('Located a total of %i runs to analyze in this directory.\n', int32(numel(dlist)));
            
            if nargin < 3; A = RHD_Analyzer(); end
            
            a0 = A.automaticMode;
            A.automaticMode = 1;
            
            q0 = self.dangerous_autoOverwrite;
            self.dangerous_autoOverwrite = 1;
            
            for dlc = startat:numel(dlist)
                cd(dlist(dlc).name);
                A.dbgclear(); % for safety
                A.runFullAnalysis();
                %cd ..;
                
                fprintf('Finished %i/%i\n', int32(dlc), int32(numel(dlist)));
            end
            
            A.automaticMode = a0;
            self.dangerous_autoOverwrite = q0;
        end
        
        function tf = checkSelfConsistency(self)
             % tf = checkSelfConsistency()
             % checks that all elements of the internal point-cloud,
             % (peakFreqs, peakMassAmps, peakLumAmps, machPts, thetaPts, fnormPts,xnormPts,
             % radnormPts), are the same size.
             % Returns true if self consistent, screams bloody murder and returns false otherwise
             
             tf = 1;
             s0 = size(self.peakFreqs,1);
             if size(self.peakMassAmps,1) ~= s0
                 warning('ALERT: mass amplitude size inconsistent!'); tf = 0;
             end
             if size(self.peakLumAmps, 1) ~= s0
                 warning('ALERT: luminance amplitude size inconsistent!'); tf = 0;
             end
             if size(self.machPts,2)  ~= s0
                 warning('ALERT: size of machPts inconsistent!'); tf = 0;
             end
             if size(self.thetaPts, 2) ~= s0
                 warning('ALERT: size of thetaPts inconsistent!'); tf = 0;
             end
             if size(self.xnormPts,2)  ~= s0
                 warning('ALERT: size of xnormPts inconsistent!'); tf = 0;
             end
             if size(self.radnormPts,2) ~= s0
                 warning('ALERT: size of radnormPts inconsistent!'); tf = 0;
             end
        end
        
        function removePoint(self, M, theta)
            % removePoint(self, M, theta)
            % Checks if I have a datapoint for (M, theta) and removes it
            % from all fields if so
            samem = find(abs(self.machPts - M) < 1e-12);
            samet = find(abs(self.thetaPts-theta) < 1e-12);
            
            if ~isempty(intersect(samem, samet))
                % this is a duplicate point
                p = intersect(samem, samet);
                
                self.printPoint(p);
                if self.dangerous_autoOverwrite ~= 1
                    really = input('This point exists in my dataset. Input 314 if sure you want to remove: ');
                else
                    really = 314;
                end
                
                if really == 314
                    n = size(self.machPts,1);
                    s = [1:(p-1) (p+1):n];
                    
                    self.machPts = self.machPts(s);
                    self.thetaPts = self.thetaPts(s);
                    
                    self.peakFreqs = self.peakFreqs(s,:);
                    self.peakMassAmps = self.peakMassAmps(s,:);
                    self.peakLumAmps = self.peakLumAmps(s,:);
                    
                    self.freqPts = self.freqPts(s);
                    self.modePts = self.modePts(s);
                    
                    self.fnormPts = self.fnormPts(s);
                    self.xnormPts = self.xnormPts(s);
                    self.radnormPts = self.radnormPts(s);
                    
                    self.fallbackRate = self.fallbackRate(s);
                    self.nShockCells = self.nShockCells(s);
                    self.spectralResolution = self.spectralResolution(s);
                end
            end
        end
        
        function insertPointNew(self, M, theta, data, numprops)
            % FMHandler.insertPointNew(M, theta, data, [conv level, fallback rate, n shock cells, spec resoln])
            % M: mach
            % theta: radiation theta
            % data: 11x3 block, Nth row has [frequency, x amp, radiance amp] of (n-1)th mode
            % convLvl: Convergence level (scale of 1, crap, to 5, golden); 0 if not given
            samem = find(abs(self.machPts - M) < 1e-12);
            samet = find(abs(self.thetaPts-theta) < 1e-12);
            
            p = intersect(samem, samet);
            
            if nargin < 5; numprops = [0 0 0 0]; end
            
            if ~isempty(p)
                % this is a duplicate point
                if self.dangerous_autoOverwrite ~= 1
                    fprintf('WARNING: This point already in internal dataset with conv lvl = %i\nEnter 314 to overwrite with point with conv lvl=%i: ', int32(self.convergenceLevel(p)), int32(numprops(1)));
                    really = input(': ');
                else
                    really = 314;
                end
                
                if really == 314
                    self.machPts(p) = M;
                    self.thetaPts(p) = theta;
                    
                    self.peakFreqs(p, :) = data(:, 1)';
                    self.peakMassAmps(p, :) = data(:, 2)';
                    self.peakLumAmps(p, :) = data(:, 3)';
                    
                    
                    self.convergenceLevel(p) = numprops(1);
                    self.fallbackRate(p) = numprops(2);
                    self.nShockCells(p) = numprops(3);
                    self.spectralResolution(p) = numprops(4);
                end
            else
                self.machPts(end+1) = M;
                self.thetaPts(end+1) = theta;

                self.peakFreqs(end+1, :) = data(:, 1)';
                self.peakMassAmps(end+1, :) = data(:, 2)';
                self.peakLumAmps(end+1, :) = data(:, 3)';
 
                self.convergenceLevel(end+1) = numprops(1);
                self.fallbackRate(end+1) = numprops(2);
                self.nShockCells(end+1) = numprops(3);
                self.spectralResolution(end+1) = numprops(4);
                
                h = HDJumpSolver(self.machPts(end), 0, self.gamma);
                R = RadiatingFlowSolver(h.rho(2), h.v(1,2), 0, 0, 0, h.Pgas(2), self.gamma, 1, self.thetaPts(end), 1.05);
                xshock = R.calculateFlowTable();
                self.fnormPts(end+1) = h.v(1,1) / xshock;
                self.xnormPts(end+1) = xshock;
                self.radnormPts(end+1) = R.luminance;
            end
            
            if ~isempty(self.dataFilename)
                save([self.dataDir '/' self.dataFilename], 'self');
            end
            
        end
        
        function printPoint(self, idx)
            fprintf('Mach = %f, theta=%f ||| freq=%f, mode=%i\n', self.machPts(idx), self.thetaPts(idx), self.freqPts(idx), self.modePts(idx));
        end
        
        function importAnotherFMHandler(self, other)
            % importAnotherFMHandler(other)
            % read all data points out of 'other' and insertPointNew() them to myself
            % be careful about setting self.dangerous_autoOverwrite!
            
            if ~isa(other, 'FMHandler2')
                disp('Did not receive an FMHandler2: Cannot import.');
                return;
            end
            
            if self.checkSelfConsistency() ~= true
                error('I am not self consistent: aborting import.');
            end
            if other.checkSelfConsistency() ~= true
                error('Other FMHandler2 is not self consistent: aborting import.');
            end
            
            if other.gamma ~= self.gamma
                um = input('The FMHandlers have different gamma values. Input 314 if you are absolutely sure. This is probably a terrible idea: ');
                if um ~= 314; return; end
            end
            
            nOther = numel(other.machPts);
            for N = 1:nOther
                m = other.machPts(N);
                t = other.thetaPts(N);
                data = [other.peakFreqs(N,:)' other.peakMassAmps(N,:)' other.peakLumAmps(N,:)'];
                np = [other.convergenceLevel(N), other.fallbackRate(N), other.nShockCells(N), other.spectralResolution(N)];
                
                p = self.findPoint(m, t);
                if p > 0 %[conv level, fallback rate, n shock cells, spec resoln]
                    if self.convergenceLevel(p) > other.convergenceLevel(N)
                        fprintf('For point (%f, %f), existing point with conv lvl %i > new %i. Ignoring new.\n', m, t, self.convergenceLevel(p), other.convergenceLevel(N));
                        continue;
                    end
                end
                
                self.insertPointNew(m, t, data, np);
            end
        end

        function rebuildFnorms(self, type)
            % rebuildFnorms(type) 
            % Recomputes the equilibrium radiating flow for all stored
            % parameters and recomputes the frequency/amplitude
            % normalization coefficients.
            % if type == 1 or empty, uses standard normalization
            % if type == 2, uses flattening normalization.
            
            if nargin < 2; type = 1; end

            self.fnormPts = zeros(size(self.machPts));
            for q = 1:numel(self.machPts)
                h = HDJumpSolver(self.machPts(q), 0, self.gamma);
                R = RadiatingFlowSolver(h.rho(2), h.v(1,2), 0, 0, 0, h.Pgas(2), self.gamma, 1, self.thetaPts(q), 1.05);
                xshock = R.calculateFlowTable();
                
                if type == 1
                    xi = 1;
                    self.fNormMode = 1;
                else
                    switch round(100*self.gamma)
                        case 167
                            %xi = (1 + 1.75 / self.machPts(q)) * (1 - .025*self.thetaPts(q));
                            xi = (1 + .7/(self.machPts(q)-1)) * (1 - .025*self.thetaPts(q));
                            %xi = (self.machPts(q)) * (1 - .025*self.thetaPts(q)) / (self.machPts(q) - 1);
                        case 140
                            xi = (1 + 2.50 / self.machPts(q)) * (1 - .040*self.thetaPts(q));
                        case 129
                            xi = (1 + 2.84 / self.machPts(q)) * (1 - .060*self.thetaPts(q));
                    end
                    self.fNormMode = 2;
                end
                self.fnormPts(q) = xi * h.v(1,1) / xshock;
                self.xnormPts(q) = xshock;
                self.radnormPts(q) = R.luminance;
                waitbar(q/numel(self.machPts));
            end
        end
        
        function updateConvergenceLevel(self, m, t, lvl)
            % updateConvergenceLevel(m, t, lvl)
            % Updates the simulation's stored value for the 'convergence
            % level' of the simulation. This rates the quality of the
            % results on a qualitative scale from 1 (definitely not the
            % final state) to 5 (definitely converged, flawless spectrum
            % with excellent resolution)
            p = self.findPoint(m, t);
            
            if p > 0
                self.convergenceLevel(p) = lvl;
            else
                disp('No such point!\n');
            end
        end
        
        function S = queryAt(self, m, t)
            % S = queryAt(Mach, theta)
            % Returns data for the input (M, t) point. Data is linearly
            % extrapolated inside of known data points
            ff = scatteredInterpolant(self.machPts, self.thetaPts, 2*pi*self.freqPts ./ self.fnormPts);
            ff.Method = 'linear';
            ff.ExtrapolationMethod = 'none';
            
            freq = ff(m, t);
            
            ff = scatteredInterpolant(self.machPts, self.thetaPts, self.modePts);
            ff.Method = 'nearest';
            ff.ExtrapolationMethod = 'none';
            
            modes = ff(m, t);
            
            S = struct('freq', freq, 'mode', modes);
        end
        
        function p = findPoint(self, m, t)
            % p = self.findPoint(Mach, theta)
            % looks for a data point matching the input (m, t) value and
            % returns the index within the FMHandler's internal data
            % structures if found, or -1 if not.
            a = find(self.machPts == m);
            b = find(self.thetaPts == t);

            p = intersect(a, b);
            
            if numel(p) == 0
                p = -1;
            end
        end

        function tf = havePoint(self, m, t)
            % tf = havePoint(Mach, theta)
            % returns true if a data point with the given parameters is
            % stored and false if not.
            a = find(self.machPts == m);
            b = find(self.thetaPts == t);

            if numel(intersect(a,b)) > 0
                tf = true;
            else
                tf = false;
            end

        end

        function l = findUnanalyedRuns(self)
            % self.findUnanalyzedRuns() searches for any radiating shock
            % directory name (RADHD_*) in the cwd whose gamma matches my
            % gamma and displays any for which we lack a data point.

            d = dir('RADHD*');

            for q = 1:numel(d)
                p = RHD_utils.parseDirectoryName(d(q).name);

                if p.gamma ~= round(100*self.gamma); continue; end

                tau = self.havePoint(p.m, p.theta);
                if tau == false
                    disp(d(q).name)
                end
            end
        end
        
        function R = selectRunsByConvergenceLevel(self, lvl)
            % Searches for all RADHD* directories in the pwd
            % and lists returns the list of all which have convergence level below lvl.
            d = dir('RADHD*');
            
            fprintf('There are %i radiating shock runs in the cwd.\n', int32(numel(d)));
            
            R = cell([numel(d) 1]);
            nR = 1;
            
            for N = 1:numel(d)
                 dn = d(N).name;
                 props = RHD_utils.parseDirectoryName(dn);
                 
                 if props.gamma == round(100*self.gamma)
                    p = self.findPoint(props.m, props.theta);
                    if p > 0
                        if self.convergenceLevel(p) == lvl
                            R{nR} = dn;
                            nR = nR + 1;
                        end
                    end
                 end
            end
           
            fprintf('Found %i runs in cwd that report convergence level of %i with my gamma\n', int32(nR-1), int32(lvl));
            
            R = R(1:(nR-1));
        end
        
        function plist = selectParamsByConvergenceLevel(self, lvl)
            ll = find(self.convergenceLevel <= lvl);
            machs = self.machPts(ll);
            thets = self.thetaPts(ll);
            falls = zeros(size(machs));
            
            plist = [machs' thets' falls'];
        end
        
        function outlist = pruneParamlistByConvergenceLvl(self, plist, critLevel)
            
            N = size(plist, 1);
            xlist = ones([N 1]);
            for x = 1:N
                p = self.findPoint(plist(x,1), plist(x,2));
                if p > 0
                    if self.convergenceLevel(p) >= critLevel
                        xlist(x) = 0;
                    end
                end
            end
            
            outlist = plist(find(xlist), :);
        end

        function emitDominantFreqPlot(self, qty, logScale, colorBy)
            % .emitDominantFreqPlot(self, qty, logScale, colorBy)
            %   qty: 1 = frequency, default;  2 = x amplitude; 3 = luminance amplitude
            %        4 = rms delta-luminosity; 5 = fallback rate; 6 = # shock cells;
            %        7 = spectral resolution
            %   logScale: If true, rendered in log scale. Otherwise, linear (default)
            %   colorBy: 1 = dominant mode (default); 2 = frequency; 3 = x amp; 4 = lum amp,
            %            5 = convergence quality; 6 = fallback speed; 7 = # shock cells; 
            %            8 = spectral resolution

            if nargin < 4; colorBy = 1; end
            if nargin < 3; logScale = 0; end
            if nargin < 2; qty = 1; end
            
            n = size(self.peakMassAmps,1);
            
            [~, idx] = max(self.peakMassAmps, [], 2);
            
            q = (1:n)' + n*(idx-1);
            
            switch qty
                case 1
                    z = 2*pi*self.peakFreqs(q) ./ self.fnormPts(:);
                    titlestring = 'z: \omega; ';
                    zstring = '\omega';
                case 2
                    z = self.peakMassAmps(q) ./ self.xnormPts(:);
                    titlestring = 'z: \delta x; ';
                    zstring = '\delta x / x_{shock}';
                case 3
                    z = self.peakLumAmps(q);% ./ self.radnormPts(:);
                    titlestring = 'z: \delta L; ';
                    zstring = '\delta L / L_{eq}';
                case 4
                    z = sqrt(sum(self.peakLumAmps.^2, 2));
                    titlestring = 'z: rms \delta L; ';
                    zstring = 'rms(\delta L) / L_{eq';
                case 5
                    z = self.fallbackRate';
                    titlestring = 'z: fallback speed';
                    zstring = 'V_{fallback}';
                case 6
                    z = self.nShockCells';
                    titlestring = 'z: # shock cells';
                    zstring = 'N cells';
                case 7
                    z = self.spectralResolution';
                    titlestring = 'z: Spectral resolution';
                    zstring = 'd\omega';
                otherwise
                    error('Invalid qty argument: not 1 to 7');
            end
            
            switch colorBy
                case 1 % dominant position modulation's mode #
                    [~, c] = max(self.peakMassAmps, [], 2);
                    c = c - 1;
                    zl = [0 8];
                    titlestring = [titlestring 'color: mode #'];
                case 2
                    c = 2*pi*self.peakFreqs(q)./ self.fnormPts(:);
                    zl = [0 5];
                    titlestring = [titlestring 'color: \omega'];
                case 3
                    c = self.peakMassAmps(q)./ self.xnormPts(:);
                    zl = [0 .5];
                    titlestring = [titlestring 'color: \delta x'];
                case 4
                    c = self.peakLumAmps(q);
                    zl = [0 1];
                    titlestring = [titlestring 'color: \delta L'];
                case 5
                    c = self.convergenceLevel';
                    zl = [0 5];
                    titlestring = [titlestring 'color: convergence lvl'];
                case 6
                    c = self.fallbackRate';
                    zl = [0 .03];
                    titlestring = [titlestring 'color: fallback speed'];
                case 7
                    c = self.nShockCells';
                    titlestring = [titlestring 'color: # shock cells'];
                    zl = [0 max(c)];
                case 8
                    c = self.spectralResolution';
                    titlestring = [titlestring 'color: Spectral resolution'];
                    zl = [0 max(c)];
                otherwise
                    error('colorBy is not one of 1-8.');
            end
                
            dohaveit = (z ~= 0);
           
            if logScale; z = log(z); zl = log(zl + .00001); end
   
            u = self.MachRange; v = self.thetaRange;
            [m, t] = ndgrid(u(1):u(2):u(3), v(1):v(2):v(3));
                     
            ff = scatteredInterpolant(self.machPts(dohaveit)', self.thetaPts(dohaveit)', z(dohaveit));
            ff.Method = 'linear';
            ff.ExtrapolationMethod = 'none';
            
            zsmooth = ff(m, t);
            
            fmode = scatteredInterpolant(self.machPts(dohaveit)', self.thetaPts(dohaveit)', c(dohaveit));
            fmode.Method = 'linear';
            fmode.ExtrapolationMethod = 'none';
            
            csmooth = fmode(m, t);
            
            surf(v(1):v(2):v(3), u(1):u(2):u(3), zsmooth, csmooth);
            hold on;
            scatter3(self.thetaPts(dohaveit)', self.machPts(dohaveit)', z(dohaveit), 'r*');
            
            xlabel('\theta');
            ylabel('Mach');
            zlabel(zstring);
            title(titlestring);
            hold off;
            view([-123 32]);
            
            colormap('jet');
            ca = gca();
            ca.CLim = zl;
            colorbar;
        end
        
        function generate3DPlot(self, drawOverlaid, qty, logScale, colorBy)
            % .generate3DPlot(self, drawOverlaid, qty, logScale, colorBy)
            %   drawOverlaid: if < 0, plots all modes (F through 10O).
            %                 Otherwise plots this mode. A vector of modes is acceptable too.
            %                 default, -1
            %   qty: 1 = frequency, default;  2 = x amplitude; 3 = luminance amplitude
            %   logScale: If true, rendered in log scale. Otherwise, linear (default)
            %   colorBy: 1 = dominant mode (default); 2 = frequency; 3 = x
            %   amp; 4 = lum amp, 5 = convergence level, 6 = fallback rate, 7 = # shock cells,
            %        8 = spectral resolution
            
            if nargin < 5; colorBy = 1; end
            if nargin < 4; logScale = 0; end
            if nargin < 3; qty = 1; end
            if nargin < 2; drawOverlaid = -1; end
            
            % default to all modes
            if drawOverlaid < 0; drawOverlaid = 0:10; end
            
            u = self.MachRange; v = self.thetaRange;
            [m, t] = ndgrid(u(1):u(2):u(3), v(1):v(2):v(3));
            
            colstring = 'kbcgyrkbcgyr';
            
            for modeno = 1:numel(drawOverlaid)
                q = drawOverlaid(modeno)+1;
                switch qty
                    case 1
                        z = 2*pi*self.peakFreqs(:, q) ./ self.fnormPts(:);
                    case 2
                        z = self.peakMassAmps(:, q) ./ self.xnormPts(:);
                    case 3
                        z = self.peakLumAmps(:, q);% ./ self.radnormPts(:);
                    otherwise
                        error('Invalid qty argument: not 1, 2 or 3');
                end
                
                switch colorBy
                    case 1 % dominant position modulation's mode #
                        [~, c] = max(self.peakMassAmps, [], 2);
                        c = c - 1;
                    case 2
                        c = 2*pi*self.peakFreqs(:, q);
                    case 3
                        c = self.peakMassAmps(:, q);
                    case 4
                        c = self.peakLumAmps(:, q);
                    case 5
                        c = self.convergenceLevel';
                    case 6
                        c = self.fallbackRate';
                    case 7
                        c = self.nShockCells';
                    case 8
                        c = self.spectralResolution';
                    otherwise
                        error('colorBy is not one of 1, 2, 3, or 4.');
                end
                
                
                dohaveit = (z ~= 0);
                z(~dohaveit) = NaN;
                dohaveit = 1:numel(self.machPts);
                if numel(find(dohaveit)) < 3; continue; end
                
                if logScale; z = log(z); end
                
                ff = scatteredInterpolant(self.machPts(dohaveit)', self.thetaPts(dohaveit)', z(dohaveit));
                ff.Method = 'linear';
                ff.ExtrapolationMethod = 'none';
                
                zsmooth = ff(m, t);
                
                fmode = scatteredInterpolant(self.machPts(dohaveit)', self.thetaPts(dohaveit)', c(dohaveit));
                fmode.Method = 'linear';
                fmode.ExtrapolationMethod = 'none';

                if 0
                    figure(2);
                    hold off;
                    tau = delaunay(self.machPts(dohaveit)', self.thetaPts(dohaveit)');
                    trisurf(tau, self.thetaPts, self.machPts, self.freqPts, self.modePts)
                    hold on;
                    scatter3(self.thetaPts, self.machPts, self.freqPts, 'r*');
                    xlabel('\theta');
                    ylabel('Mach');
                    zlabel('F');
                    hold off;
                    view([-123 32]);
                    
                    colormap('jet');
                    ca = gca();
                    ca.CLim = [0, 6];
                    colorbar;
                    figure(1);
                end

                csmooth = fmode(m, t);
                
                surf(v(1):v(2):v(3), u(1):u(2):u(3), zsmooth, csmooth);
                hold on;
                scatter3(self.thetaPts(dohaveit)', self.machPts(dohaveit)', z(dohaveit), [colstring(modeno) '*']);
            end
            
            
            xlabel('\theta');
            ylabel('Mach');
            zlabel('F');
            hold off;
            view([-123 32]);
            
            colormap('jet');
            ca = gca();
            ca.CLim = [0, 6];
            colorbar;
        end        
        
        function searchForDuplicateRuns(self, directs)
            % FMHandler2.searchForDuplicateRuns(directories) searches all radiative shock runs
            % in these directories with my gamma for any that have the same M and same theta. If
            % directories is empty, uses pwd() only. Otherwise provide a {'/list', '/of', '/dirs'}
            
            if nargin < 1; directs = {pwd()}; end
            
            r = zeros(size(self.machPts));
            
            d0 = pwd();
            d=[];
            for N = 1:numel(directs)
                cd(d0);
                cd(directs{N});
                d = [d; dir(sprintf('RAD*gam%i', int32(round(100*self.gamma))))];
            end
            
            for n = 1:numel(d)
                x = RHD_utils.parseDirectoryName(d(n).name);
                p = self.findPoint(x.m, x.theta);
                
                if p > 0
                    r(p) = r(p) + 1;
                end
            end
            
            np = 0;
            for n = 1:numel(r)
                if r(n) > 1
                    fprintf('For parameters M=%f theta=%f found %i runs:\n', self.machPts(n), self.thetaPts(n), r(n));
                    
                    for N = 1:numel(directs)
                        cd(d0);
                        cd(directs{N});
                        fprintf(directs{N}); fprintf('  ');
                        self.searchMode(self.machPts(n), self.thetaPts(n));
                        
                    end
                    %self.searchMode(self.machPts(n), self.thetaPts(n));
                    np = np + 1;
                    if np == 10
                        fprintf('10 shown: continue printing? ', numel(r));
                        tf = input('');
                        if ~tf; break; end
                    end
                end
            end
            
            cd(d0);
        end
        
        function searchMode(self, M, th)
            if nargin < 3
                th = M(:,2);
                M =  M(:,1);
                for n = 1:size(M,1)
                    self.searchMode(M(n), th(n));
                end
            else
                str = sprintf('!ls -l | grep ms%i_ | grep radth%i_ | grep gam%i\n', M, round(100*th), round(100*self.gamma));
                eval(str);
            end
        end
        
        
        function fbr = queryFallback(self, m, theta)
            fsi = scatteredInterpolant(self.machPts', self.thetaPts', self.fallbackRate');
            fbr = fsi(m, theta);
        end
        
        
        
    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                      P R O T E C T E D    [M]
        
    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %                                                 S T A T I C    [M]
        
    end%PROTECTED
    
end%CLASS
