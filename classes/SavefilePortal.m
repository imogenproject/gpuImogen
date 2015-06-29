classdef SavefilePortal < handle
    %___________________________________________________________________________________________________
    
    %===================================================================================================
    properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
    end %PUBLIC
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
        savefileDirectory;
        currentFrame;
        typeToLoad;

        savefileList;
        strnames={'X','Y','Z','XY','XZ','YZ','XYZ'};

	directoryStack;
    end %PROTECTED
    
    %===================================================================================================
    methods %                                                                     G E T / S E T  [M]
    end%GET/SET
    
    %===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
        function self = SavefilePortal(wd)
            % SavefilePortal(directory) provides a unified way to access
            % all Imogen save data
            if nargin == 0; wd = pwd(); end

            self.changeDirectory(wd);
            self.rescanDirectory();
            self.typeToLoad = 7;
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
	    if id < 1; id = 7; disp('Invalid ID; defaulted to XYZ (7)'); end
	    if id > 7; id = 7; disp('Invalid ID; defaulted to XYZ (7)'); end
            self.typeToLoad = id;
        end
        function accessX(self); self.setFrametype(1); end
        function accessY(self); self.setFrametype(2); end
        function accessZ(self); self.setFrametype(3); end
        function accessXY(self); self.setFrametype(4); end
        function accessXZ(self); self.setFrametype(5); end
        function accessYZ(self); self.setFrametype(6); end
        function accessXYZ(self); self.setFrametype(7); end

        % Next/previous/start/last to make raw setFrame() friendlier
        function [F glitch] = nextFrame(self)
            % F = nextFrame() returns the next Imogen saveframe of the
            % currently selected type
            n = self.currentFrame(self.typeToLoad) + 1;
            [F glitch] = self.setFrame(n);
        end

        function F = previousFrame(self)
            % F = previousFrame() returns the previous Imogen saveframe of
            % the current type
            n = self.currentFrame(self.typeToLoad)-1;
            [F glitch] = self.setFrame(n);
        end
        
        function [F glitch] = jumpToFirstFrame(self)
            % Resets the current frame to the first
    	    [F glitch] = self.setFrame(1);
        end
        
        function [F glitch] = jumpToLastFrame(self)
           % Hop to the last frame available
           n = self.numFrames();
           [F glitch] = self.setFrame(n);
        end
        
        function [F glitch] = setFrame(self, f)
            % F = setFrame(n) jumps to the indicated frame of the current
            % type; Automatically clamps to [1 ... #frames]
            glitch = 0; % assume no problem...
            if f < 1; f = 1; glitch = -1; end
            if f >= self.numFrames();
                f = self.numFrames();
                glitch = 1;
            end
            
            b = getfield(self.savefileList,self.strnames{self.typeToLoad});
            self.currentFrame(self.typeToLoad) = f;
            
            self.pushdir(self.savefileDirectory);
            F = util_LoadWholeFrame(self.typeToLoad, self.savefileList.misc.padlen, b(f));
            self.popdir();
        end
            
       function arewe = atLastFrame(self)
           % true if the portal is currently aimed at the last frame of the
           % current type, otherwise false
            arewe = (self.currentFrame(self.typeToLoad) == self.numFrames());
       end

       function n = tellFrame(self)
           % N = tellFrame() indicates which frame was the last
           % loaded/returned
            n = self.currentFrame(self.typeToLoad);
       end

	function n = numFrames(self)
        % n = numFrames() returns how many frames of the current type are
        % accessible in the current directory
	    n = numel(getfield(self.savefileList,self.strnames{self.typeToLoad}));
	end

        function rescanDirectory(self)
            self.pushdir(self.savefileDirectory);
            self.savefileList = enumerateSavefiles();
            self.popdir();
        end

    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                      P R O T E C T E D    [M]
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