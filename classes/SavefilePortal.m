classdef SavefilePortal < handle
    %___________________________________________________________________________________________________
    
    %===================================================================================================
    properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = public) %                           P U B L I C  [P]
        numFrames;
        currentType;
        varFormat;
    end %PUBLIC
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
        savefileDirectory;
        currentFrame;
        typeToLoad;

        savefileList;
        strnames={'X','Y','Z','XY','XZ','YZ','XYZ'};

        % the default, code-generated prefixes used to name the 7 save output types
        standardNamePrefixes = {'1D_X', '1D_Y', '1D_Z', '2D_XY', '2D_XZ', '2D_YZ', '3D_XYZ'};
        
        directoryStack;

        pParallelMode;
        pMetamode;
    end %PROTECTED
    
    %===================================================================================================
    methods %                                                                     G E T / S E T  [M]
    end%GET/SET
    
    %===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
        function self = SavefilePortal(wd, ttl)
            % SavefilePortal(directory) provides a unified way to access
            % all Imogen save data
            if nargin == 0; wd = pwd(); end
            
            self.numFrames = 0;

            if nargin > 1
                self.setFrametype(ttl);
            else
                self.typeToLoad = 7;
            end
            
            self.changeDirectory(wd);
            self.rescanDirectory();
            
            self.setParallelMode(0);
            self.setMetamode(0);
            self.setVarFormat('default');
        end

        function setParallelMode(self, enable)
            if enable
                self.pParallelMode = 1;
                if mpi_isinitialized() == 0
                    warning('SavefilePortal:setParallelMode, parallel mode turned on but MPI not initialized? calling mpi_init()...');
                    mpi_init();
                end
            else
                self.pParallelMode = 0;
            end
        end

        function setMetamode(self, m)
            if m; self.pMetamode = 'metaonly'; else; self.pMetamode = ''; end 
        end
        
        function setVarFormat(self, f)
            if strcmp(f, 'default')
                self.varFormat = 'default';
                return;
            end
            if strcmp(f, 'conservative')
                self.varFormat = 'conservative';
                return;
            end
            if strcmp(f, 'primitive')
                self.varFormat = 'primitive';
                return;
            end
            
            fprintf('WARNING: SavefilePortal.setVarFormat(''%s'') is not valid: no change from %s\n', f, self.varFormat);
        end

        function changeDirectory(self, nwd)
            % changeDirectory(nwd) makes the portal operate on the New
            % Working Directory; Resets the portal (all frames -> first).
            % Does not corrupt caller's PWD.
            self.savefileDirectory = nwd;
            
            self.pushdir(nwd);
            self.currentFrame = [0 0 0 0 0 0 0];
            self.popdir();
        end

        function setFrametype(self, id)
        % setFrametype(id) sets the frame used to X, Y, Z, XY, XZ, YZ, XYZ
        % for id values 1-7 respectively.
            if isa(id, 'double')
                problem = 0;
                if id < 1; problem = id; id = 7; disp('Invalid ID; defaulted to XYZ (7)'); end
                if id > 7; problem = id; id = 7; disp('Invalid ID; defaulted to XYZ (7)'); end
                if problem
                    fprintf('Got id of %i, valid values are 1 (X), 2 (Y), 3 (Z), 4 (XY), 5 (XZ), 6 (YZ), 7 (XYZ)\n', int32(problem));
                end
            else
                if strcmp(id,'X');   id = 1; end
                if strcmp(id,'Y');   id = 2; end
                if strcmp(id,'Z');   id = 3; end
                if strcmp(id,'XY');  id = 4; end
                if strcmp(id,'XZ');  id = 5; end
                if strcmp(id,'YZ');  id = 6; end
                if strcmp(id,'XYZ'); id = 7; end
                if isa(id,'double') == false
                    disp('Received the following for ID:\n');
                    disp(id)
                    disp('Valid string values are X, Y, Z, XY, XZ, YZ, XYZ (case sensitive)\n -> Defaulting to XYZ <-\n');
                    id = 7;
                end
            end

            self.typeToLoad = id;
            self.updatePublicState();
        end
        
        function accessX(self); self.setFrametype(1); end
        function accessY(self); self.setFrametype(2); end
        function accessZ(self); self.setFrametype(3); end
        function accessXY(self); self.setFrametype(4); end
        function accessXZ(self); self.setFrametype(5); end
        function accessYZ(self); self.setFrametype(6); end
        function accessXYZ(self); self.setFrametype(7); end

        function IC = returnInitializer(self) %#ok<STOUT>
            self.pushdir(self.savefileDirectory);
            load('SimInitializer_rank0','IC');
            self.popdir();
            return;
        end

        % Next/previous/start/last to make raw setFrame() friendlier
        function [F, glitch] = nextFrame(self, setrank)
            % F = nextFrame() returns the next Imogen saveframe of the
            % currently selected type
            n = self.currentFrame(self.typeToLoad) + 1;
            if nargin < 2
                [F, glitch] = self.setFrame(n);
            else
                [F, glitch] = self.setFrame(n, setrank);
            end
        end

        function [F, glitch] = previousFrame(self, setrank)
            % F = previousFrame() returns the previous Imogen saveframe of
            % the current type
            n = self.currentFrame(self.typeToLoad)-1;
            if nargin < 2
                [F, glitch] = self.setFrame(n);
            else
                [F, glitch] = self.setFrame(n, setrank);
            end
        end
        
        function [F, glitch] = jumpToFirstFrame(self, setrank)
            % Resets the current frame to the first
            if nargin < 2
                [F, glitch] = self.setFrame(1);
            else
                [F, glitch] = self.setFrame(1, setrank);
            end
        end
        
        function [F, glitch] = jumpToLastFrame(self, setrank)
           % Hop to the last frame available
           n = self.numFrames;
           if nargin < 2
                [F, glitch] = self.setFrame(n);
            else
                [F, glitch] = self.setFrame(n, setrank);
            end
        end
        
        function [F, glitch] = setFrame(self, f, fixedrank)
            % F = setFrame(n) jumps to the indicated frame of the current
            % type; Automatically clamps to [1 ... #frames]
            glitch = 0; % assume no problem...
            if f < 1; f = 1; glitch = -1; end
            if f >= self.numFrames
                f = self.numFrames;
                glitch = 1;
            end
            
            b = self.savefileList.(self.strnames{self.typeToLoad});
            self.currentFrame(self.typeToLoad) = f;
            
            % Figure out what, if any, rank in particular we want to load
            r = -1;
            if self.pParallelMode
                r = mpi_myrank();
            elseif nargin == 3
                r = fixedrank;
            end

            if strcmp(self.pMetamode, 'metaonly'); mo = 1; else; mo = 0; end
            
            % Figure out if we only want meta, a particular rank file, or
            % the whole enchilada
            self.pushdir(self.savefileDirectory);
            if (r >= 0) || mo
                if r < 0; r = 0; end
		if mo
                    F = util_LoadFrameSegment(self.typeToLoad, r, b(f), self.pMetamode);
		else
                    F = DataFrame(util_LoadFrameSegment(self.typeToLoad, r, b(f), self.pMetamode));
		end
            else
                F = DataFrame(util_LoadWholeFrame(self.typeToLoad, b(f)));
            end
            
            self.popdir();
        end
            
        function met = getMetadata(self, f, rankToLoad)
            
            if nargin < 3; rankToLoad = 0; end
            if nargin < 2; f = self.currentFrame(self.typeToLoad); end
            
            m = self.pMetamode; self.pMetamode = 'metaonly';
            met = self.setFrame(f, rankToLoad);
            self.pMetamode = m;
            
            if f < 1; f = 1; glitch = -1; end
            if f >= self.numFrames
                f = self.numFrames;
                glitch = 1;
            end
        end
        
        function fname = getSegmentFilename(self, frameno, rankno)
            if nargin < 3; rankno = 0; end
            if nargin < 2; frameno = self.currentFrame(self.typeToLoad); end
            
            b = self.savefileList.(self.strnames{self.typeToLoad});
            
            self.pushdir(self.savefileDirectory);
           [act, fname] = util_FindSegmentFile(self.typeToLoad, rankno, b(frameno)); 
            self.popdir();
        end

        function arewe = atLastFrame(self)
            % true if the portal is currently aimed at the last frame of the
            % current type, otherwise false
            arewe = (self.currentFrame(self.typeToLoad) == self.numFrames);
        end

        function n = tellFrame(self)
            % N = tellFrame() indicates which frame was the last
            % loaded/returned
            n = self.currentFrame(self.typeToLoad);
        end


        function rescanDirectory(self)
            self.pushdir(self.savefileDirectory);
            self.savefileList = enumerateSavefiles();
            
            self.updatePublicState();
            
            self.popdir();
        end
        
        function S = getIniSettings(self)
            S = load([self.savefileDirectory  '/ini_settings.mat']);
        end
        
        function S = getInitialConditions(self)
            if self.pParallelMode == 0
                r = 0;
            else
                r = mpi_myrank();
            end
            f = sprintf('/SimInitializer_rank%i.mat', int32(r));
            u = load([self.savefileDirectory f], 'IC');
            S = u.IC;
        end

        function n = getFilenamePrefix(self)
            n = self.standardNamePrefixes{self.typeToLoad};
        end
        
    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                      P R O T E C T E D    [M]
        function updatePublicState(self)
            % n = numFrames returns how many frames of the current type are
            % accessible in the current directory. Set the type using the
            % access* functions, or directly with .setFrametype(1...7)
            if ~isempty(self.savefileList)
                self.numFrames = numel(self.savefileList.(self.strnames{self.typeToLoad}));
            end
            self.currentType = self.strnames{self.typeToLoad};
        end

        function pushdir(self, D)
            self.directoryStack{end+1} = pwd();
            cd(D);
        end

        function popdir(self)
            cd(self.directoryStack{end});
            self.directoryStack = self.directoryStack(1:(end-1));
        end
    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %                                                 S T A T I C    [M]

        function y = cvtToPrimitive(x)
            y = x;
            if isfield(y, 'velX'); return; end % nothing to be done
            
            minv = 1./x.mass;

            psq = x.momX.*x.momX;
            y.velX = x.momX.*minv;
            y = rmfield(y, 'momX');

            psq = psq + x.momY.*x.momY;
            y.velY = x.momY.*minv;
            y = rmfield(y, 'momY');

            psq = psq + x.momZ.*x.momZ;
            y.velZ = x.momZ.*minv;
            y = rmfield(y, 'momZ');

            y.eint = x.ener - .5*psq.*minv;
            y = rmfield(y, 'ener');

            if isfield(x, 'mass2')
                minv = 1./x.mass2;

                psq = x.momX2.*x.momX2;
                y.velX2 = x.momX2.*minv;
                y = rmfield(y, 'momX2');

                psq = psq + x.momY2.*x.momY2;
                y.velY2 = x.momY2.*minv;
                y = rmfield(y, 'momY2');

                psq = psq + x.momZ2.*x.momZ;
                y.velZ2 = x.momZ2.*minv;
                y = rmfield(y, 'momZ2');

                y.eint2 = x.ener2 - .5*psq.*minv;
                y = rmfield(y, 'ener2');
            end

        end
        

        function y = cvtToConservative(x)
            y = x;
            if isfield(y, 'momX'); return; end % nothing to be done

            vsq = y.velX.*y.velX;
            y.momX = y.velX.*y.mass;
            y = rmfield(y, 'velX');

            vsq = vsq + y.velY.*y.velY;
            y.momY = y.velY.*y.mass;
            y = rmfield(y, 'velY');

            vsq = vsq + y.velZ.*y.velZ;
            y.momZ = y.velZ.*y.mass;
            y = rmfield(y, 'velZ');

            y.ener = y.eint + .5*x.mass.*vsq;
            y = rmfield(y, 'eint');

            if isfield(x, 'mass2')
                vsq = x.velX2.*x.mvelX2;
                y.momX2 = x.velX2.*x.mass2;
                y = rmfield(y, 'velX2');

                vsq = vsq + x.velY2.*x.velY2;
                y.momY2 = x.velY2.*x.mass2;
                y = rmfield(y, 'velY2');

                vsq = vsq + x.velZ2.*x.velZ;
                y.momZ2 = x.velZ2.*x.mass2;
                y = rmfield(y, 'velZ2');

                y.ener2 = x.eint2 + .5*vsq.*x.mass2;
                y = rmfield(y, 'eint2');
            end
        end

    end%PROTECTED
    
end%CLASS
