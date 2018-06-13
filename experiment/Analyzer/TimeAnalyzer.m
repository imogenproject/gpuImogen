classdef TimeAnalyzer < handle

    properties (Constant = true)
        version = 1.0;
    end

    properties(SetAccess = public, GetAccess = public)

    end

    properties(SetAccess = protected, GetAccess = public)
        nFrames;
        is2d;   % True if run was of z extent 1, otherwise false.

        originalSrcDirectory; % Set at analyze time, the directory the analyzed data resided in.
        

    end

    properties(SetAccess = protected, GetAccess = protected)
        inputBasename;
        inputPadlength;
        inputFrameRange;
        maxFrameno;

    end

methods % SET

end

methods (Access = public)
    function object = TimeAnalysis(self)
        object.nFrames = 0;
        object.is2d    = NaN;

        object.originalSrcDirectory = pwd();
    end



    function selectFileset(obj, basename, framerange)
        if nargin ~= 4
            obj.inputBasename  = input('Base filename for source files, (e.g. "3D_XYZ", no trailing _):','s');
            obj.inputFrameRange       = input('Range of frames to export; _START = 0 (e.g. 0:50:1000 to do every 50th frame from start to 1000): ');
        else
            obj.inputBasename   = basename;
            obj.inputFrameRange = framerange;
        end
%    timeNormalization = input('Characteristic time to normalize by (e.g. alfven crossing time or characteristic rotation period. If in doubt hit enter): ');
%    if timeNormalization == 0; timeNormalization = 1; end;

        if max(round(obj.inputFrameRange) - obj.inputFrameRange) ~= 0; error('ERROR: Frame obj.inputFrameRange is not integer-valued.\n'); end
        if min(obj.inputFrameRange) < 0; error('ERROR: Frame obj.inputFrameRange must be nonnegative.\n'); end

        obj.inputFrameRange = obj.removeNonexistantEntries(obj.inputBasename, obj.inputFrameRange);
        obj.maxFrameno = max(obj.inputFrameRange);
        obj.nFrames = numel(obj.inputFrameRange);

        obj.originalSrcDirectory = pwd();
    end


    function output = runAnonOnFrames(obj, afunc)
        output = [];

        for iter = 1:obj.nFrames
            dataframe = util_LoadWholeFrame(obj.inputBasename, obj.inputFrameRange(iter));
            output{iter} = afunc(dataframe);
        end
    end

    function squashed = cellsToMatrix(obj, input)
        squashed = [];
        for q = 1:numel(input)
            squashed(:,:,q) = input{q};
        end
    end

end

methods (Access = protected)

    function newframeranges = removeNonexistantEntries(obj, namebase, frameranges)

        existframeranges = [];

        for ITER = 1:numel(range)
            ftype = util_FindSegmentFile(namebase, 0, range(ITER));

            if ftype > 0; existframeranges(end+1) = ITER; end
        end

        newframeranges = frameranges(existframeranges);
        if numel(newframeranges) ~= numel(frameranges)
            fprintf('WARNING: Removed %i entries that could not be opened from list.\n', numel(frameranges)-numel(newframeranges));
        end

        if numel(newframeranges) == 0
            error('FATAL: No files indicated existed. Perhaps need to remove _ from base name?');
        end

    end

end % protected methods;

methods (Static = true)
       function help()
        fprintf('I am a TimeAnalyzer base class; I help sift through a directory full of Imogen savefiles and generate useful time series data. My basic functions are:\n\tselectFileset(basename, framerange) - select (interactively if I don''t get all 3) what frames to load\n\toutput = runAnonOnFrames(anonymous function) - return a cell array produced by passing dataframes in sequence to the anonymous function which may return whatever it pleases.\n\tsquashed = cellsToMatrix(cells) - takes cells from above and, if they are nice 2x2 matrices, returns a 3d matrix with time stacked in the 3rd dimension.\n\n');
    end 
end

end % class
