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
        
        convergenceLevel;
    end %PUBLIC
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = public) %                P R O T E C T E D [P]
        machPts; % X
        thetaPts; % Y
        
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

        function autoanalyzeEntireDirectory(self, startat)
            % autoanalyzeEntireDirectory(self, startat)
            % Searches the cwd for any directories matching RADHD*gam###
            % where ### is the adiabatic index of myself x 100 and
            % and runs RHD_1danalysis in them with rhdAutodrive=2 (auto
            % everything except for spectral analysis interval)
            if nargin < 2; startat = 1; end
            dlist = dir(sprintf('RADHD*gam%3i',round(100*self.gamma)));
            
            rhdAutodrive = 1; % FULL SPEED AHEAD!!! *if possible
            
            fprintf('Located a total of %i runs to analyze in this directory.\n', int32(numel(dlist)));
            
            for dlc = startat:numel(dlist)
                cd(dlist(dlc).name);
                RHD_1danalysis
                cd ..
                fprintf('Finished %i/%i\n', int32(dlc), int32(numel(dlist)));
            end
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
                end
            end
        end
        
        function insertPointNew(self, M, theta, data, convLvl)
            % FMHandler.insertPointNew(M, theta, data, convLvl)
            % M: mach
            % theta: radiation theta
            % data: 11x3 block, Nth row has [frequency, x amp, radiance amp] of (n-1)th mode
            % convLvl: Convergence level (scale of 1, crap, to 5, golden); 0 if not given
            samem = find(abs(self.machPts - M) < 1e-12);
            samet = find(abs(self.thetaPts-theta) < 1e-12);
            
            p = intersect(samem, samet);
            
            if nargin < 5; convLvl = -1; end
            
            if ~isempty(p)
                % this is a duplicate point
                if self.dangerous_autoOverwrite ~= 1
                    fprintf('WARNING: This point already in internal dataset with conv lvl = %i\nEnter 314 to overwrite with point with conv lvl=%i', int32(self.convergenceLevel(p)), int32(convLvl));
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
                    if convLvl > -1
                        self.convergenceLevel(p) = convLvl;
                    end
                end
            else
                self.machPts(end+1) = M;
                self.thetaPts(end+1) = theta;

                self.peakFreqs(end+1, :) = data(:, 1)';
                self.peakMassAmps(end+1, :) = data(:, 2)';
                self.peakLumAmps(end+1, :) = data(:, 3)';

                self.convergenceLevel(end+1) = convLvl;
                
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
                
                p = self.findPoint(m, t);
                if p > 0
                    if self.convergenceLevel(p) > other.convergenceLevel(N)
                        fprintf('For point (%f, %f), existing point with conv lvl %i > new %i. Ignoring new.\n', m, t, self.convergenceLevel(p), other.convergenceLevel(N));
                        continue;
                    end
                end
                
                self.insertPointNew(m, t, data);
                self.updateConvergenceLevel(m, t, other.convergenceLevel(N));
            end
        end

        function rebuildFnorms(self)
            % rebuildFnorms() 
            % Recomputes the equilibrium radiating flow for all stored
            % parameters and recomputes the frequency/amplitude
            % normalization coefficients

            self.fnormPts = zeros(size(self.machPts));
            for q = 1:numel(self.machPts)
                h = HDJumpSolver(self.machPts(q), 0, self.gamma);
                R = RadiatingFlowSolver(h.rho(2), h.v(1,2), 0, 0, 0, h.Pgas(2), self.gamma, 1, self.thetaPts(q), 1.05);
                xshock = R.calculateFlowTable();
                self.fnormPts(q) = h.v(1,1) / xshock * (1 + 2.84 / self.machPts(q)) * (1 - .06 * self.thetaPts(q));
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

        function emitDominantFreqPlot(self, qty, logScale, colorBy)
            % .emitDominantFreqPlot(self, qty, logScale, colorBy)
            %   qty: 1 = frequency, default;  2 = x amplitude; 3 = luminance amplitude
            %   logScale: If true, rendered in log scale. Otherwise, linear (default)
            %   colorBy: 1 = dominant mode (default); 2 = frequency; 3 = x amp; 4 = lum amp,
            %            5 = convergence quality
            
            if nargin < 4; colorBy = 1; end
            if nargin < 3; logScale = 0; end
            if nargin < 2; qty = 1; end
            
            n = size(self.peakMassAmps,1);
            
            [~, idx] = max(self.peakMassAmps, [], 2);
            
            q = (1:n)' + n*(idx-1);
            
            switch qty
                case 1
                    z = 2*pi*self.peakFreqs(q) ./ self.fnormPts(:);
                case 2
                    z = self.peakMassAmps(q) ./ self.xnormPts(:);
                case 3
                    z = self.peakLumAmps(q);% ./ self.radnormPts(:);
                case 4
                    z = sqrt(sum(self.peakLumAmps.^2, 2));
                otherwise
                    error('Invalid qty argument: not 1 to 4');
            end
            
            switch colorBy
                case 1 % dominant position modulation's mode #
                    [~, c] = max(self.peakMassAmps, [], 2);
                    c = c - 1;
                case 2
                    c = 2*pi*self.peakFreqs(q);
                case 3
                    c = self.peakMassAmps(q);
                case 4
                    c = self.peakLumAmps(q);
                case 5
                    c = self.convergenceLevel';
                otherwise
                    error('colorBy is not one of 1, 2, 3, or 4.');
            end
                
            dohaveit = (z ~= 0);
           
            if logScale; z = log(z); end
            
   
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
            zlabel('F');
            hold off;
            view([-123 32]);
            
            colormap('jet');
            ca = gca();
            ca.CLim = [0, 6];
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
            %   amp; 4 = lum amp, 5 = convergence level
            
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
        
        function searchForDuplicateRuns(self)
           
            r = zeros(size(self.machPts));
            
            d = dir(sprintf('RAD*gam%i', int32(round(100*self.gamma))));
            
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
                    self.searchMode(self.machPts(n), self.thetaPts(n));
                    np = np + 1;
                    if np == 10
                        fprintf('10 shown: continue printing? ', numel(r));
                        tf = input('');
                        if ~tf; break; end
                    end
                end
            end
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
    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                      P R O T E C T E D    [M]
        
    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %                                                 S T A T I C    [M]
        
    end%PROTECTED
    
end%CLASS
