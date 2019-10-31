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
        
        peakFreqs;
        peakMassAmps;
        peakLumAmps;
    end %PUBLIC
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = public) %                P R O T E C T E D [P]
        machPts; % X
        thetaPts; % Y
        
        fnormPts;   % Coefficients for frequency normalization
        xnormPts;   % X_shock coefficients for position amplitude normalization
        radnormPts; % Equilibrium radiance for luminance amplitude normalization
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
        
        function insertPointNew(self, M, theta, data)
            % FMHandler.insertPointNew(M, theta, data)
            % M: mach
            % theta: radiation theta
            % data: 11x3 block, Nth row has [frequency, x amp, radiance amp] of (n-1)th mode
            samem = find(abs(self.machPts - M) < 1e-12);
            samet = find(abs(self.thetaPts-theta) < 1e-12);
            
            p = intersect(samem, samet);
            
            if ~isempty(p)
                % this is a duplicate point
                if self.dangerous_autoOverwrite ~= 1
                    really = input('WARNING: this point already exists in my internal dataset.\nEnter 314 if you are sure: ');
                else
                    really = 314;
                end
                
                if really == 314
                    self.machPts(p) = M;
                    self.thetaPts(p) = theta;
                    
                    self.peakFreqs(p, :) = data(:, 1)';
                    self.peakMassAmps(p, :) = data(:, 2)';
                    self.peakLumAmps(p, :) = data(:, 3)';
                end
            else
                self.machPts(end+1) = M;
                self.thetaPts(end+1) = theta;
                
                self.peakFreqs(end+1, :) = data(:, 1)';
                self.peakMassAmps(end+1, :) = data(:, 2)';
                self.peakLumAmps(end+1, :) = data(:, 3)';
                
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
            
            %otherPts = 
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
                
                self.insertPointNew(m, t, data);
            end
        end

        function rebuildFnorms(self)
            u = self.MachRange; v = self.thetaRange;
            
            [m, t] = ndgrid(u(1):u(2):u(3), v(1):v(2):v(3));
            
            self.fnormPts = zeros(size(self.machPts));
            for q = 1:numel(self.machPts)
                h = HDJumpSolver(self.machPts(q), 0, self.gamma);
                R = RadiatingFlowSolver(h.rho(2), h.v(1,2), 0, 0, 0, h.Pgas(2), self.gamma, 1, self.thetaPts(q), 1.05);
                xshock = R.calculateFlowTable();
                self.fnormPts(q) = h.v(1,1) / xshock;% * (1 + 1.05*self.gamma / self.machPts(q)) * (1 - .038 * self.thetaPts(q));
                waitbar(q/numel(self.machPts));
            end
            
            self.fnormPts = reshape(self.fnormPts, [numel(self.machPts) 1]);
        end
        
        function S = queryAt(self, m, t)
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

        function tf = havePoint(self, m, t)
            a = find(self.machPts == m);
            b = find(self.thetaPts == t);

            if numel(intersect(a,b)) > 0
                tf = true;
            else
                tf = false;
            end

        end

        function l = findUnanalyedRuns(self)

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

        function emitDominantFreqPlot(self, qty, logScale, colorBy)
            % .emitDominantFreqPlot(self, qty, logScale, colorBy)
            %   qty: 1 = frequency, default;  2 = x amplitude; 3 = luminance amplitude
            %   logScale: If true, rendered in log scale. Otherwise, linear (default)
            %   colorBy: 1 = dominant mode (default); 2 = frequency; 3 = x amp; 4 = lum amp
            
            if nargin < 4; colorBy = 1; end
            if nargin < 3; logScale = 0; end
            if nargin < 2; qty = 1; end
            
            n = size(self.peakMassAmps,1);
            
            [~, idx] = max(self.peakMassAmps, [], 2);
            
            q = (1:n)' + n*(idx-1);
            
            freq = self.peakFreqs(q) ./ self.fnormPts(:);

            switch qty
                case 1
                    z = 2*pi*self.peakFreqs(q) ./ self.fnormPts(:);
                case 2
                    z = self.peakMassAmps(q) ./ self.xnormPts(:);
                case 3
                    z = self.peakLumAmps(q);% ./ self.radnormPts(:);
                otherwise
                    error('Invalid qty argument: not 1, 2 or 3');
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
        %{
        What do we want?
        - pictures of dominant luminance
        - frequency over params
        - mode amplitude over params (overlaid)
        - lum amplitude over params (overlaid)
        - linear or log vertical scale
        
        so, plot params:
        (dominant or overlaid)
        (frequency, X amplitude, luminance amplitude) for z
        (linear or log scale for Z)
        (dom mode #, frequency, X amp, luminance amp) for color
        %}
        
        
        function generate3DPlot(self, drawOverlaid, qty, logScale, colorBy)
            % .generate3DPlot(self, drawOverlaid, qty, logScale, colorBy)
            %   drawOverlaid: if < 0, plots all modes (F through 10O).
            %                 Otherwise plots this mode. A vector of modes is acceptable too.
            %                 default, -1
            %   qty: 1 = frequency, default;  2 = x amplitude; 3 = luminance amplitude
            %   logScale: If true, rendered in log scale. Otherwise, linear (default)
            %   colorBy: 1 = dominant mode (default); 2 = frequency; 3 = x amp; 4 = lum amp
            
            if nargin < 5; colorBy = 1; end
            if nargin < 4; logScale = 0; end
            if nargin < 3; qty = 1; end
            if nargin < 2; drawOverlaid = -1; end
            
            % default to all modes
            if drawOverlaid < 0; drawOverlaid = 0:10; end
            
            u = self.MachRange; v = self.thetaRange;
            [m, t] = ndgrid(u(1):u(2):u(3), v(1):v(2):v(3));
            
            colstring = 'rgbcmkrgbcmk';
            
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
