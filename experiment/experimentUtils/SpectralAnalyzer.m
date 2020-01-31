classdef SpectralAnalyzer < handle
    % Class annotation template for creating new classes.
    %___________________________________________________________________________________________________
    
    %===================================================================================================
    properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
        
    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        table;
        binWidth;
    end %PUBLIC
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
    end %PROTECTED
    
    %===================================================================================================
    methods %                                                                     G E T / S E T  [M]
    end%GET/SET
    
    %===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
        
        function self = SpectralAnalyzer(peaks, luminance)
            st = struct('f',[], 'fsigma', [], 'foffset', Inf, 'dx', [], 'dl', [], 'm', -2, 'hd', [], 'imd', []);
            
            self.table = st;
            for n = 1:size(peaks,1)
                self.table(n) = st;
                self.table(n).f = peaks(n, 1);
                self.table(n).fsigma = peaks(n, 3);
                self.table(n).dx = peaks(n, 2);
                if nargin > 1; self.table(n).dl = luminance(n,2); end
            end
            
        end
        
        function aprioriTag(self, tagfunc)
            % Accepts a function handle @tagfunc(f) that returns a mode #
            for n = 1:numel(self.table)
                if self.table(n).m ~= -2; continue; end
                
                [self.table(n).m self.table(n).foffset] = tagfunc(self.table(n).f);
                
                if self.table(n).m > -1
                    self.table(n).hd = 1;
                    self.tagHarmonics(n);
                else
                    self.table(n).m = -2;
                end
            end
        end
        
        function tagHarmonics(self, m0)
            f0 = self.table(m0).f;
            df0 = self.table(m0).fsigma;
            
            for n = 1:numel(self.table)
                if n == m0; continue; end
                if self.table(n).m ~= -2; continue; end
                
                fUnc = self.rms([df0 self.table(n).fsigma]);
                
                if abs( (self.table(n).f / f0) - round(self.table(n).f / f0) ) < 2*fUnc
                    self.table(n).m = self.table(m0).m;
                    self.table(n).hd = round(self.table(n).f / f0);
                    self.table(n).foffset = self.table(n).f - self.table(n).hd * f0;
                end
            end
        end
        
        function untag(self, m)
            self.table(m).m = -2;
            self.table(m).hd = [];
            self.table(m).imd = [];
        end
        
        function thereCanBeOnlyOne(self)
           
            m0 = [];
            for n = 1:numel(self.table)
                if self.table(n).m >= 0
                    m0(end+1) = self.table(n).m;
                end
            end
            m0 = unique(m0);
            
            for a = 1:numel(m0)
                c = [];
                for b = 1:numel(self.table)
                   if (self.table(b).m == m0(a)) && (self.table(b).hd == 1)
                       c(end+1) = b;
                   end
                end

                if numel(c) > 1
                    fprintf('Multiple tones marked as mode %i; Keeping only largest dx amplitude & retagging...\n', m0(a));
                    amax = 0; aidx = -1;
                    
                    for b = 1:numel(c)
                        if self.table(c(b)).dx > amax
                            amax = self.table(c(b)).dx;
                            aidx = b;
                        end
                    end
                    
                    % dump all tags on this mode
                    for b = 1:numel(self.table)
                        if self.table(b).m == m0(a)
                            self.untag(b);
                        end
                    end
                    
                    % retag only the largest
                    self.table(c(aidx)).m = m0(a);
                    self.table(c(aidx)).hd = 1;
                    self.tagHarmonics(c(aidx));
                end
            end
                
            
        end
        
        function testFakeIntermodulation(self)
          
            % gather a list of all base tones & their 2nd harmonics if available
            % Note that this will include the hypothetical fake harmonic
            tlist = [];
            btones = [];
            for n = 1:numel(self.table)
                if self.table(n).m > -1
                    if self.table(n).hd == 1
                        tlist(end+1,:) = [ self.table(n).f, self.table(n).fsigma, self.table(n).foffset, n];
                    end
                    if self.table(n).hd == 1
                        btones(end+1,:) = [self.table(n).f, self.table(n).fsigma, self.table(n).foffset, n];
                    end
                end
            end
            
            % compare all 2nd and 3rd order intermodulation terms with all base tones
            [imma, immb] = ndgrid([-2 -1 1 2], [-2 -1 1 2]);
            
            notmodes = [];
            
            for a = 1:size(tlist,1)
                for b = (a+1):size(tlist,1)
                    % all [-2 -1 1 2] x Fa + [-2 -1 1 2] x Fb combinations
                    tones = imma*tlist(a,1) + immb*tlist(b,1);
                    
                    % compare with all detected base tones
                    for c = 1:size(btones,1)
                        poffset = abs(tones - btones(c,1));
                        
                        % check for those closer than the frequency difference vs the a priori tag
                        [row, col] = find(abs(poffset) < abs(btones(c,3)));
                        if numel(row) > 0
%                            fprintf('Found %i low-order intermodulations closer to base tone F=%f than a priori \n', numel(row), btones(c,1));
                            
                            % find the best one
                            % this is probably the stupid way to do this...
                            k = 1;
                            for beta = 2:numel(row)
                                if abs(poffset(row(beta), col(beta)) < abs(poffset(row(k), col(k)))); k = beta; end
                            end
                            
                            % This always runs both ways (since A + B = C can be solved 3 ways)
                            
                            % Based on the assumption of strict perturbative ordering (harmonics
                            % and higher orders always less than base tones and lower orders),
                            % pick only the solution yielding the low amplitude one
                            if (self.table(btones(c,4)).dx < self.table(tlist(a,4)).dx) && (self.table(btones(c,4)).dx < self.table(tlist(b,4)).dx)
  %                              fprintf('Closest: %ix%f + %ix%f = %f (err=%f) vs a priori %f\n', imma(row(k), col(k)), tlist(a,1), immb(row(k), col(k)), tlist(b,1), btones(c,1), poffset(row(k), col(k)), btones(c,3));
                                notmodes(end+1) = btones(c,4);
                            end
                        end
                    end
                    
                    %fproposed = abs(tlist(a,1) - tlist(b,1));
                end
            end

            % avoid multiple entries if multiple forms of IMD create the same fake tone
            notmodes = unique(notmodes);
            
            for n = 1:numel(notmodes)
                tau = self.table(notmodes(n));
                
                for a = 1:numel(self.table)
                    if self.table(a).m == tau.m
                        self.untag(a);
                    end
                end
            end
        end
        
        function tagIntermodulation(self)
            % gather a list of all base tones & their harmonics
            tlist = [];
            % and a list of unsorted tones
            btones = [];
            for n = 1:numel(self.table)
                if self.table(n).m > -1
                    if self.table(n).hd == 1
                        tlist(end+1,:) = [ self.table(n).f, self.table(n).fsigma, self.table(n).foffset, n];
                    end
                end
                if self.table(n).m == -2
                    btones(end+1,:) = [self.table(n).f, self.table(n).fsigma, self.table(n).foffset, n];
                end
            end
            
            % compare all 2nd and 3rd order intermodulation terms with all base tones
            [imma, immb] = ndgrid([-3 -2 -1 1 2 3], [-3 -2 -1 1 2 3]);
            
            for a = 1:size(tlist,1)
                for b = (a+1):size(tlist,1)
                    % all [-2 -1 1 2] x Fa + [-2 -1 1 2] x Fb combinations
                    tones = imma*tlist(a,1) + immb*tlist(b,1);
                    
                    % compare with all detected base tones
                    for c = 1:size(btones,1)
                        poffset = abs(tones - btones(c,1));
                        
                        % check for those closer than the frequency difference vs the a priori tag
                        [row, col] = find(abs(poffset) < self.rms([tlist(a,2) tlist(b,2) btones(c,2)]));
                        if numel(row) > 0
%                            fprintf('Found %i low-order intermodulations close to unknown tone F=%f\n', numel(row), btones(c,1));
                            
                            % find the best one
                            % this is probably the stupid way to do this...
                            k = 1;
                            for beta = 2:numel(row)
                                if abs(poffset(row(beta), col(beta)) < abs(poffset(row(k), col(k)))); k = beta; end
                            end
                            
                            % Based on the assumption of strict perturbative ordering (harmonics
                            % and higher orders always less than base tones and lower orders),
                            % pick only the solution yielding the low amplitude one
                            if (self.table(btones(c,4)).dx < self.table(tlist(a,4)).dx) && (self.table(btones(c,4)).dx < self.table(tlist(b,4)).dx)
 %                               fprintf('Closest: %ix%f + %ix%f = %f (err=%f)\n', imma(row(k), col(k)), tlist(a,1), immb(row(k), col(k)), tlist(b,1), btones(c,1), poffset(row(k), col(k)));
                                self.table(btones(c, 4)).m = -1; % IMD marker
                                self.table(btones(c, 4)).hd = abs(imma(row(k),col(k))) + abs(immb(row(k),col(k)));
                                self.table(btones(c, 4)).imd = [imma(row(k),col(k)), tlist(a,4), immb(row(k),col(k)), tlist(b,4)];
                                self.table(btones(c, 4)).foffset = poffset(row(k),col(k));
                            end
                        end
                    end
                    
                end
            end
            
        end
        
        function printTable(self)
            
            fprintf('\n');
            for n = 1:numel(self.table)
                tau = self.table(n);
                switch tau.m
                    case -2; tstr = '??'; idst = '';
                    case -1
                        tstr = sprintf('IMD%i', tau.hd);
                        idst = sprintf('%ix%s + %ix%s', tau.imd(1), RHD_utils.assignModeName(self.table(tau.imd(2)).m), tau.imd(3), RHD_utils.assignModeName(self.table(tau.imd(4)).m));
                    case 0
                        tstr = 'F   ';
                        if tau.hd > 1
                            idst = sprintf('%iHD', tau.hd);
                        else
                            idst = 'Shock mode';
                        end
                    otherwise
                        tstr = sprintf('%iO  ', self.table(n).m);
                        if tau.hd > 1
                            idst = sprintf('%iHD', tau.hd);
                        else
                            idst = 'Shock mode';
                        end
                end
                
                fprintf('f = %f | mag %f | %s | %s\n', self.table(n).f, self.table(n).dx, tstr, idst);
            end
        end
    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                      P R O T E C T E D    [M]
    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %                                                 S T A T I C    [M]
        function y = rms(x)
             y = sqrt(sum(x.^2)) / numel(x);
        end
        
    end%PROTECTED
    
end%CLASS
