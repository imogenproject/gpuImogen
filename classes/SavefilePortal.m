classdef SavefilePortal < handle
    %___________________________________________________________________________________________________
    
    %===================================================================================================
    properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = public) %                           P U B L I C  [P]
        numFrames;
        currentType;
    end %PUBLIC
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
        savefileDirectory;
        currentFrame;
        typeToLoad;

        savefileList;
        strnames={'X','Y','Z','XY','XZ','YZ','XYZ'};

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
            if nargin == 2
                [F, glitch] = self.setFrame(n);
            else
                [F, glitch] = self.setFrame(n, setrank);
            end
        end

        function [F, glitch] = previousFrame(self, setrank)
            % F = previousFrame() returns the previous Imogen saveframe of
            % the current type
            n = self.currentFrame(self.typeToLoad)-1;
            if nargin == 2
                [F, glitch] = self.setFrame(n);
            else
                [F, glitch] = self.setFrame(n, setrank);
            end
        end
        
        function [F, glitch] = jumpToFirstFrame(self, setrank)
            % Resets the current frame to the first
            if nargin == 2
                [F, glitch] = self.setFrame(1);
            else
                [F, glitch] = self.setFrame(1, setrank);
            end
        end
        
        function [F, glitch] = jumpToLastFrame(self, setrank)
           % Hop to the last frame available
           n = self.numFrames;
           if nargin == 2
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
            
            r = -1;
            if self.pParallelMode
                r = mpi_myrank();
            elseif nargin == 3
                r = fixedrank;
            end
            
            self.pushdir(self.savefileDirectory);
            if (r >= 0) || (strcmp(self.pMetamode, 'metaonly'))
                if r < 0; r = 0; end
                F = util_LoadFrameSegment(self.typeToLoad, r, b(f), self.pMetamode);
            else
                F = util_LoadWholeFrame(self.typeToLoad, b(f));
            end
            self.popdir();
        end
            
        function met = getMetadata(self, f, rankToLoad)
            glitch = 0; % assume no problem...
            
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
    end%PROTECTED
    
end%CLASS
