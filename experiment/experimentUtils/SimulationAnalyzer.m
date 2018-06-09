classdef SimulationAnalyzer < handle

    properties (Constant = true)
        version = 1.0;
    end

    properties(SetAccess = private, GetAccess = public)
        originalSrcDirectory; % Set at analyze time, the directory the analyzed data resided in.
    end

    properties(SetAccess = public, GetAccess = public)

        frameT; % Simple cartesian coordinates
        frameX; frameY; frameZ;
        kxValues; kyValues; kzValues;
        
        equil;
        analyzedFrames;
        setOfFrames;

        analyzerFields;
    end

    properties(SetAccess = protected, GetAccess = protected)
        inputBasename;
        inputFrameRange;
        maxFrameno;
        
        frameAnalyzerFunction; % function to do static analysis on a frame
        %simAnalyzerFunction;   % function to examine whole data set
        verbose;
    end

methods % SET

end

methods (Access = public)
    function serialize(obj, filename) % Because save('filename','results') is just too
                                      % darn easy to simply work
                                      % Or because Matlab can't handle really large files :(
        FILE = fopen(filename,'w');

        fwrite(FILE, size(obj.pre.drho), 'double');
        fwrite(FILE, numel(obj.originalSrcDirectory), 'double');
        fwrite(FILE, obj.originalSrcDirectory, 'char*1');
        fwrite(FILE, [numel(obj.gridXvals) size(obj.front.X,2) size(obj.front.X,1)], 'double');
        
        fwrite(FILE, obj.frameTimes, 'double');

        fwrite(FILE, double(obj.frameLinearity), 'double');

        fwrite(FILE, numel(obj.gridXvals), 'double');
        fwrite(FILE, obj.gridXvals, 'double');

        fwrite(FILE, obj.kyValues, 'double');
        fwrite(FILE, obj.kyWavenums, 'double');
        
        fwrite(FILE, obj.kzValues, 'double');
        fwrite(FILE, obj.kzWavenums, 'double');

        fwrite(FILE, numel(obj.linearFrames), 'double');
        if numel(obj.linearFrames) > 0
            fwrite(FILE, obj.linearFrames, 'double');
        end

        S = serdes; % serializer/deserializer helper class

        S.writeStructure(FILE, obj.equil);
        S.writeStructure(FILE, obj.front);
	S.writeStructure(FILE, obj.pre);
	S.writeStructure(FILE, obj.post);
	if numel(obj.linearFrames) > 0; S.writeStructure(FILE, obj.omega); end

        fclose(FILE);

    end

    function deserialize(obj, filename)
        FILE = fopen(filename,'r');
        x = fread(FILE, 5, 'double'); % [#ky #kz nx nt strlen(source dir)]

        obj.originalSrcDirectory = char(fread(FILE, x(5), 'char*1')');

        obj.nModes = [x(1) x(2)];
        obj.nFrames = x(4);

        nxAnalysis = x(3);

        if obj.nModes(2) == 1; obj.is2d = 1; else; obj.is2d = 0; end

        y = fread(FILE, 3, 'double');
        nxsim = y(1);
        nysim = y(2);
        nzsim = y(3);

        obj.frameTimes = fread(FILE, [1 obj.nFrames], 'double');

        y = fread(FILE, obj.nFrames, 'double');
        obj.frameLinearity = logical(reshape(y,[1 obj.nFrames]));

        nxsim = fread(FILE, 1, 'double');
        obj.gridXvals = fread(FILE, nxsim, 'double');

        obj.kyValues = fread(FILE, obj.nModes(1), 'double');
        obj.kyWavenums = fread(FILE, [1 obj.nModes(1)], 'double');

        obj.kzValues = fread(FILE, obj.nModes(2), 'double');
        obj.kzWavenums = fread(FILE, [1 obj.nModes(2)], 'double');

        y = fread(FILE, 1, 'double');
        if y(1) > 0
            obj.linearFrames = fread(FILE, y, 'double');
            obj.linearFrames = obj.linearFrames';
        end

	S = serdes;

        obj.equil = S.readStructure(FILE);
        obj.front = S.readStructure(FILE);
        obj.pre   = S.readStructure(FILE);
        obj.post  = S.readStructure(FILE);
        if numel(obj.linearFrames) > 0; obj.omega = S.readStructure(FILE); end

        fclose(FILE);
    end

    function obj = SimulationAnalyzer(basename, framerange, verbosity)
        if nargin < 3; error('Require: SimulationAnalyzer(base filename, [range:of:frames], verbose=1 or 0=off\n'); end
        
        obj.setBaseInputs(basename);

        obj.analyzerFields = {'frameT'};
        
        obj.setOfFrames = framerange;
        obj.verbose = verbosity;
    end
    
    function setAnalyzerFunction(obj, func); obj.frameAnalyzerFunction = func; end
    
    function addFields(obj, fields)
        obj.analyzerFields = {obj.analyzerFields{:} fields{:}};
    end

    function setBaseInputs(obj, basename)
        if nargin == 2
            obj.inputBasename = basename;
        else
            obj.inputBasename = input('Please input base name (e.g. 2D_XY): ', 's');
        end
        
        % As a courtesy, strip a trailing _ without whining
        if obj.inputBasename(end) == '_'; obj.inputBasename = obj.inputBasename(1:(end-1)); end

        obj.originalSrcDirectory = pwd();
    end
    
    function setEquilibriumFrame(obj, eqnumber)
        obj.equil = util_LoadWholeFrame(obj.inputBasename, eqnumber);
        
        obj.frameX = cumsum(1:size(obj.equil.mass,1))*obj.equil.dGrid{1};
        obj.frameY = cumsum(1:size(obj.equil.mass,2))*obj.equil.dGrid{2};
        obj.frameZ = cumsum(1:size(obj.equil.mass,3))*obj.equil.dGrid{3};

        obj.kyvalues
    end
    
    function addFramesToAnalysis(obj, newFrames)
        % Adds additional frames to be analyzed at a future time
        obj.setOfFrames = unique(sort([obj.setOfFrames newFrames]));
    end
    
    
    function updateAnalysis(obj, anames)
        % Compares analyzedFrames and setOfFrames and inserts blanks in the analyzed data sets to
        % make space for new data
        if all( size(obj.setOfFrames) == size(obj.analyzedFrames) )
            disp('No frames added since last analysis; Returning...\n');
            return;
        end

        outarray = zeros(size(obj.setOfFrames));
        lasthit = 0;
        
        % Determine array time-axis remap
        for iter = 1:numel(obj.analyzedFrames);
            for iterB = (lasthit+1):numel(obj.setOfFrames)
               if obj.setOfFrames(iterB) == obj.analyzedFrames(iter); break; end 
            end
            outarray(iterB) = iter;
            lasthit = iterB;
        end

        outcopy = find(outarray > 0);

        obj.checkFramesExist();

        originalDir = pwd();
        cd(obj.originalSrcDirectory);

        if obj.verbose; fprintf('%i *s (25/line):\n',numel(obj.setOfFrames)-numel(obj.analyzedFrames)); end

        if numel(outcopy) == 0; % No pre-existing analysis; Just run the analyzer function on all elements
            tic;
            for iter = 1:numel(outarray);

                if obj.verbose;
                    fprintf('*');
                   if mod(iter,25) == 0; fprintf('\n'); end
                end

                dataframe = util_LoadWholeFrame(obj.inputBasename, obj.setOfFrames(iter));

                % On first frame of new analysis, calculate the X/K vectors to convenientify things in the future.
                if iter == 1;
                    obj.frameX = (0:(size(dataframe.mass,1)-1))*dataframe.dGrid{1};
                    obj.frameY = (0:(size(dataframe.mass,2)-1))*dataframe.dGrid{2};
                    obj.frameZ = (0:(size(dataframe.mass,3)-1))*dataframe.dGrid{3};

                    % set K_n = 2 pi n / L
                    obj.kxValues = (0:(size(dataframe.mass,1)-1))*2*pi/(size(dataframe.mass,1)*dataframe.dGrid{1});
                    obj.kyValues = (0:(size(dataframe.mass,2)-1))*2*pi/(size(dataframe.mass,2)*dataframe.dGrid{2});
                    obj.kzValues = (0:(size(dataframe.mass,3)-1))*2*pi/(size(dataframe.mass,3)*dataframe.dGrid{3});
                end

                obj.frameT = [obj.frameT; sum(dataframe.time.history)];
                obj.FrameAnalyzer(dataframe, iter);

                if iter == 2; fprintf('\nEst. %fsec remaining in analysis.\n', (numel(outarray)-2)*toc/2); end
            end

            obj.analyzedFrames = obj.setOfFrames;
            cd(originalDir);
            return;
        end

        % In one swoop copies existing time-slice analysis data to expanded arrays
        for FIELD = 1:numel(obj.analyzerFields)
            u = size(getfield(obj, obj.analyzerFields{FIELD}));
            v = u; v(1) = numel(obj.setOfFrames);
            newf = zeros(v);

            existset = {find(outarray)};
            for j = 2:numel(u); existset{j}=1:u(j); end

            newf(existset{:}) = getfield(obj, obj.analyzerFields{FIELD});
            setfield(obj, obj.analyzerFields{FIELD}, newf);
        end

        % We now have outarray saying where to copy existing data (entry nonzero) and where
        % we need new data (entry zero) from the descendant's frame analyzer function.
        zf = 0;
        tic;
        for iter = 1:numel(obj.setOfFrames)
            if outarray(iter) > 0; continue; end
            
            zf=zf+1;
            if obj.verbose;
                fprintf('*');
                if mod(zf,25) == 0; fprintf('\n'); end
            end

            dataframe = util_LoadWholeFrame(obj.inputBasename, obj.setOfFrames(iter));
            obj.frameT(iter,1) = sum(dataframe.time.history);
            obj.frameAnalyzerFunction(dataframe, iter);

            if zf == 2; fprintf('\nEst. %fsec remaining in analysis.\n', (numel(obj.setOfFrames)-numel(obj.analyzedFrames)-2)*toc/2); end
        end

        obj.analyzedFrames = obj.setOfFrames;
        cd(originalDir);
    end
    
end % Public methods

methods % SET

end

methods (Access = protected)

    function s = framenumToFilename(obj, num)
        % This function converts a given frame number to the filaname of
	% the rank zero savefile associated with it.
	[ishere, s] = util_FindSegmentFile(obj.inputBasename, 0, num);
    end
    
    function checkFramesExist(obj)
        fexist = ones(size(obj.setOfFrames));

        disp('Checking for existence of frames to be analyzed...');

        previousDir = pwd();
        cd(obj.originalSrcDirectory);
        
        for ITER = 1:numel(obj.setOfFrames)
            fnguess = obj.framenumToFilename(obj.setOfFrames(ITER));

            doesExist = exist(fnguess, 'file');
        
            fexist(ITER) = doesExist;
        end
        
        newset = obj.setOfFrames(fexist == 2); % Perform a logical select for items which exist as files

        cd(previousDir);

        if numel(newset) == 0; error('No frames appear to exist. Leaving set untouched so you can fix what''s wrong.\n'); end
        fprintf('   Found %i/%i requested frames to exist.\n', numel(newset), numel(obj.setOfFrames));
        obj.setOfFrames = newset;
    end

end % protected methods;

end % class    
