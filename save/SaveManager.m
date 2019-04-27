classdef SaveManager < LinkedListNode
% The manager class responsible for handling saving/updating data actions.
%===================================================================================================
    properties (Constant = true, Transient = true) %                     C O N S T A N T         [P]
        SLICEFIELDS = {'x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz', 'cust'};   % Fields for slices.
        SLICELABELS = {'X', 'Y', 'Z', 'XY', 'XZ', 'YZ', 'XYZ', 'SP-XYZ'}; % Labels for slices.
    end
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public, Transient = true) %         P U B L I C  [P]
        %--- Results related logical states ---%
        saveData % Specifies whether or not to save.                                     logical
        save1DData % Specifies whether or not to save 1D slices.                         logical
        save2DData % Specifies whether or not to save 2D slices.                         logical
        save3DData% Specifies whether or not to save 3D slices.                          logical
        saveCustomData % Specifies whether or not to save custom slices.                 logical

        format; % ENUM.FORMAT_MAT or ENUM.FORMAT_NC

        done % Specifies if the code has reached the end for final save.                 logical
        previousUpdateTimes;% Saves the times of previous updates for each slice type. double(5)
                                                %        1:        last time 1D data was saved.
                                                %        2:        last time 2D data was saved.
                                                %        3:        last time 3D data was saved.
                                                %        4:        last time Custom data was saved.
                                                %        5:        last time UI was updated.
        previousUpdateWallTimes; % Same as previousUpdateTimes except for wall time     double(5)
                                 % instead of simulation time.
    
        customSaveStr % An expression to be executed for custom saving                      Str
        specialSaves1D % Array of specific iterations on which to save 1D data          int(N,1)
        specialSaves2D % Array of specific iterations on which to save 2D data          int(N,1)
        specialSaves3D % Array of specific iterations on which to save 3D data          int(N,1)

        FSAVE % Specifies if data should be saved to disk for the run.                   logical
        PERSLICE % The percentages per save for each slice type:                       double(4)
                 %        1: % between 1D data saves.
                 %        2: % between 2D data saves.
                 %        3: % between 3D data saves.
                 %        4: % between Custom data saves.
            
        ACTIVE;          % Specifies which slices should be saved.                  logical(8,1)

        SLICEINDEX;      % The indices at which to slice the grid for slice saves.      int(3,1)
        parent;          % parent manager                                          ImogenManager
        firstSave;       % Specifies if this is the first save action for the run.       logical
    end%PUBLIC

%===================================================================================================
    properties (SetAccess = private, GetAccess = private, Transient = true) %   P R I V A T E    [P]
    
    end%PRIVATE
%===================================================================================================
    methods %                                                              G E T / S E T         [M]
            
%____________________________________________________________________________________ GS: SLICEINDEX
    function set.SLICEINDEX(obj,value)
        if (ndims(value) < 3),        value = [value 1]; end
        obj.SLICEINDEX = value;
    end

    function value = get.SLICEINDEX(obj); value = obj.SLICEINDEX; end
    
%_________________________________________________________________________________________ GS: FSAVE
    function value = get.FSAVE(obj)
        value = obj.FSAVE;
    end
            
    function set.FSAVE(obj, value)
        obj.FSAVE = value;
        if (~obj.FSAVE), disp('Saving is disabled. Results in workspace only.'); end
    end
            
    end
    
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
%______________________________________________________________________________________ SaveManager
% Creates a new SaveManager instance: Presumably only called by the ImogenManager constructor
    function obj = SaveManager()
        obj = obj@LinkedListNode(); % Initialize the LL to blank

        obj.whatami = 'SaveManager';

        obj.SLICEINDEX              = ones(1,3);
        obj.ACTIVE                  = false(1,8);
        obj.previousUpdateTimes     = zeros(1,5);
        obj.previousUpdateWallTimes = zeros(1,5);
        obj.saveData                = true;
        obj.save1DData              = true;
        obj.save2DData              = true;
        obj.save3DData              = true;
        obj.saveCustomData          = false;
        obj.done                    = false;
        
        obj.format = ENUM.FORMAT_HDF;
    end
    
%_______________________________________________________________________________________ preliminary
% Handles preliminary initialization of the SaveManager after all of the initialization settings 
% have been set. This function is meant to be called by the ImogenManager only.
    function initialize(obj, IC, run, fluids, mag)
        obj.firstSave = true;
    
        % Skip if saving is inactive.
        if ~obj.FSAVE 
            return;
        end

% FIXME: This should determine *when* resultsHandler should be called
% and mark THAT iteration/time, not waste time on every iteration...
        saver = ImogenEvent([], 1, [], @resultsHandler);
        saver.armed = 1;
        run.attachEvent(saver);
        
        rez = obj.parent.geometry.globalDomainRez;
        %--- Analyze grid directions for auto-slices ---%
        [~, indexMax] = max(rez);
        [~, indexMin] = min(rez);
        
        %--- Determine auto-slices ---%
        %           If no slices were specified by the user, the manager chooses the ones that
        %           appear most appropriate for the each dimension based on grid sizes.
        noSavesSpecified = ~any(obj.ACTIVE);
        
        if ~any(obj.ACTIVE(1:3)) && noSavesSpecified
            obj.ACTIVE(indexMax) = true;
        end

        if ~any(obj.ACTIVE(4:6))&& noSavesSpecified
            switch (indexMax + indexMin)
                case 3; obj.ACTIVE(4) = true;
                case 4; obj.ACTIVE(5) = true;
                case 5; obj.ACTIVE(6) = true;
            end
        end
        
        if ~obj.ACTIVE(7) && noSavesSpecified
            obj.ACTIVE(7) = true;
        end

    end
    
    function parseIni(self, ini)
        % Reads and sets all properties defined in the initializer struct that are relevant to the
        % SaveManager object
        
        self.FSAVE = ini.save;
        self.PERSLICE(1:4) = [ini.ppSave.dim1 ini.ppSave.dim2 ini.ppSave.dim3 ini.ppSave.cust];
        
        self.format = ini.saveFormat;
        
        self.SLICEINDEX = ini.slice; % ???
        
        slLabels = {'x','y','z','xy','xz','yz','xyz','cust'};
        for i=1:8
            if ~isfield(ini.activeSlices,slLabels{i}); self.ACTIVE(i) = false;
            else; self.ACTIVE(i) = logical(ini.activeSlices.(slLabels{i}));
            end
        end
        
        % ????
        saveStr = '''slTime'',''slAbout'',''version'',''slGamma'',''sldGrid''';
        
        custom = ini.customSave;
        if isstruct(custom)
            if (isfield(custom,'mass')  && custom.mass),    saveStr = [saveStr ',''slMass''']; end
            if (isfield(custom,'mom')   && custom.mom),     saveStr = [saveStr ',''slMom'''];  end
            if (isfield(custom,'ener')  && custom.ener),    saveStr = [saveStr ',''slEner''']; end
            if (isfield(custom,'mag')   && custom.mag),     saveStr = [saveStr ',''slMag'''];  end
            self.customSaveStr = saveStr;
        else
            self.customSaveStr = '';
        end
        
        if ~isempty(ini.specSaves) && isa(ini.specSaves,'double')
            self.specialSaves3D = ini.specSaves;
            self.parent.appendInfo('Special save points 3D', self.specialSaves3D);
        else
            if isfield(ini.specSaves,'dim1')
                self.specialSaves1D = ini.specSaves.dim1;
                self.parent.appendInfo('Special save points 1D', self.save.specialSaves1D);
            end
            
            if isfield(ini.specSaves,'dim2')
                self.specialSaves2D = ini.specSaves.dim2;
                self.parent.appendInfo('Special save points 2D', self.save.specialSaves2D);
            end
            
            if isfield(ini.specSaves,'dim3')
                self.specialSaves3D = ini.specSaves.dim3;
                self.parent.appendInfo('Special save points 3D', self.save.specialSaves3D);
            end
        end
    end

%_____________________________________________________________________________________ postliminary
    function finalize(obj, run, fluids, mag)
        run.save.logPrint('SaveManager finalize called.\n');
    end

%________________________________________________________________________________ saveIniSettings
% Saves the ini structure to disk so that it can be reused later to restart a run if necessary.
    function saveIniSettings(obj, ini)

       if obj.FSAVE && ~isempty(ini) && mpi_amirank0()
           %--- Save ini Structure ---%
           %            Saves the entire ini structure to a mat file for later reloading and 
           %            reuse.
           save([obj.parent.paths.save filesep 'ini_settings.mat'],'ini');
           
            %--- Create run.log Entry ---%
            %               Writes basic run information to the run.log file for later 
            %               reference. This includes fixed basic information as well as any
            %               fields listed in the ini.logProperties.
            paths    = obj.parent.paths;
            run      = obj.parent;
            
            data     = paths.saveFolder;
            data     = [data, '\n\tSaved at: ', strrep(paths.save, '\', '/')];
            data     = [data, '\n\tVersion: ', run.detailedVersion];
            data     = [data, '\n\tStarted: ', datestr(run.time.startTime, ...
                                                       'HH:MM on mmm dd, yyyy')];
            data     = [data, '\n\tInfo: ', strrep(run.about, '---+ ', '')];

            for i=1:length(ini.logProperties)
                prop = ini.logProperties{i};
                data = [data, '\n\t', prop, ': ',  ImogenRecord.valueToString(ini.(prop))]; %#ok<AGROW>
            end
            
            data     = strcat(data, '\n');
            
            fid      = fopen(strcat(paths.results, '/run.log'), 'a');
            fprintf(fid, strcat(repmat('-', [1,80]), '\n', data));
            fclose(fid);
            
            fid      = fopen(strcat(paths.save, '/run.log'), 'w');
            fprintf(fid, data);
            fclose(fid);
       end
    end

%__________________________________________________________________________________ updateDataSaves
% Updates the data saves depending on whether the iterations or the time is expected to complete
% first.
    function updateDataSaves(obj)
        time = obj.parent.time;
        
        [~, index] = max([time.iterPercent, time.timePercent, time.wallPercent]);
        switch index
            case 1
                obj.updateIterationDataSaves(time);
            case 2
                obj.updateTimeDataSaves(time);
            case 3
                obj.updateWallDataSaves(time);
        end
        
        obj.updateOverrides(time.iteration);
        obj.scheduleSave();
    end
    
    function willsave = peekAtSave(obj)
        time = obj.parent.time;
        time.updateWallTime(); % may as well...
        
        time.fakeStep();
        [~, index] = max([time.iterPercent, time.timePercent, time.wallPercent]);
        switch index
            case 1
                willsave = max(round(time.ITERMAX*([obj.PERSLICE/100])), 1);
                willsave = min(willsave, max(floor(time.ITERMAX/4),1)); % At least 4 saves if MAXITER >= 4.
                willsave = mod(time.iteration, willsave);               % Critical loop test.
            case 2
                currentFraction = time.time / time.TIMEMAX - obj.previousUpdateTimes;
                % yes/no on whether we've passed the fraction to trigger a save for this slice
                willsave = 100*currentFraction >= [obj.PERSLICE 10];
            case 3
                willsave = (time.wallPercent - 100*obj.previousUpdateWallTimes/time.WALLMAX) ...
                    >= [obj.PERSLICE 10];
        end
        time.fakeBackstep();
        
        willsave = any(willsave);
    end
    
%__________________________________________________________________________________ updateOverrides
    function updateOverrides(obj,iteration)
        special = (obj.done || obj.firstSave);
        i       = iteration;
        
        if (~isempty(obj.specialSaves1D) && any(~any(obj.specialSaves1D - i, 1)) || special)        
            obj.save1DData = true; 
        end
                    
        if (~isempty(obj.specialSaves2D) && any(~any(obj.specialSaves2D - i, 1)) || special) 
            obj.save2DData = true; 
        end
                    
        if (~isempty(obj.specialSaves3D) && any(~any(obj.specialSaves3D - i, 1)) || special)
            obj.save3DData = true; 
        end
    end
                            
%_____________________________________________________________________________________ getSaveSlice
    function result = getSaveSlice(obj,gpuarray,type)
% This function returns the sliced data from the input array. If the array is codistributed, the 
% slice is gathered to labindex 1 and returned only to that lab. All other labs see result = [].
%>> array            Data array to be sliced.                                        double(?)
%>> type            Type of slice to be made on the array (1-8).                    int
%<< result            Resulting slice (exists only on labindex 1).                    double(?)    
% FIXME: This needs to be parallel aware. It's so not
% FIXME: This will probably require XY/XZ/YZ communicators

     N = size(gpuarray);         
        if length(N) < 3;   N = [N 1]; end
        
            switch type
                case 1;                        i = {1:N(1),obj.SLICEINDEX(2),obj.SLICEINDEX(3)};
                case 2;                        i = {obj.SLICEINDEX(1),1:N(2),obj.SLICEINDEX(3)};
                case 3;                        i = {obj.SLICEINDEX(1),obj.SLICEINDEX(2),1:N(3)};
                                    
                case 4;                        i = {1:N(1),1:N(2),obj.SLICEINDEX(3)};
                case 5;                        i = {1:N(1),obj.SLICEINDEX(2),1:N(3)};
                case 6;                        i = {obj.SLICEINDEX(1),1:N(2),1:N(3)};
                                    
                case {7, 8};        i = { 1:N(1), 1:N(2), 1:N(3) };
            end

            if isa(gpuarray,'GPU_Type'); gpuarray = gpuarray.array; end
            result = squish( gpuarray(i{:}) );
        end
            
    end%PUBLIC
    
%===================================================================================================    
methods (Access = private) %                                               P R I V A T E    [M]
%_________________________________________________________________________________ scheduleSave
% Tests save conditions and determines the state of the saveData property.
    function scheduleSave(obj)
        obj.saveData = ( obj.save1DData || obj.save2DData || obj.save3DData ...
            || obj.saveCustomData || obj.done) && obj.FSAVE;
    end
    
    
%______________________________________________________________________________ updateWallDataSaves
    function updateWallDataSaves(obj, time)
        obj.done    = (time.wallTime >= time.WALLMAX);
        wallUpdates = (time.wallPercent - 100*obj.previousUpdateWallTimes/time.WALLMAX) ...
            >= [obj.PERSLICE 10];
        
        obj.save1DData      = wallUpdates(1);
        obj.save2DData      = wallUpdates(2);
        obj.save3DData      = wallUpdates(3);
        obj.saveCustomData        = wallUpdates(4);
        
        for i=1:length(wallUpdates)
            if wallUpdates(i)
                obj.previousUpdateTimes(i)      = time.time;
                obj.previousUpdateWallTimes(i)  = time.wallTime;
            end
        end
    end
    
%_____________________________________________________________________________ updateTimeDataSaves
    function updateTimeDataSaves(obj, time)
        obj.done    = (time.time >= time.TIMEMAX);
        % What fraction of time to pass has thus far elapsed
        currentFraction = time.time / time.TIMEMAX - obj.previousUpdateTimes;
        % yes/no on whether we've passed the fraction to trigger a save for this slice
        timeUpdates = 100*currentFraction >= [obj.PERSLICE 10];
        
        vomit = 100 ./ [obj.PERSLICE 10];
        
        obj.save1DData      = timeUpdates(1);
        obj.save2DData      = timeUpdates(2);
        obj.save3DData      = timeUpdates(3);
        obj.saveCustomData  = timeUpdates(4);
        
        for i=1:length(timeUpdates)
            if timeUpdates(i)
                % Time to elapse between saves
                timePerSave = floor(time.time*vomit(i)/time.TIMEMAX) / vomit(i);
                
                obj.previousUpdateTimes(i)      = timePerSave;
                obj.previousUpdateWallTimes(i)  = time.wallTime;
            end
        end
    end
    
%_________________________________________________________________________ updateIterationDataSaves
    function updateIterationDataSaves(obj, time)
        obj.done = (time.iteration >= time.ITERMAX);
        modVal = max(round(time.ITERMAX*([obj.PERSLICE/100 0.1])), 1);
        modVal = min(modVal, max(floor(time.ITERMAX/4),1)); % At least 4 saves if MAXITER >= 4.
        modVal = mod(time.iteration, modVal);               % Critical loop test.
        
        obj.save1DData                = ~logical(modVal(1));
        obj.save2DData                = ~logical(modVal(2));
        obj.save3DData                = ~logical(modVal(3));
        obj.saveCustomData        = ~logical(modVal(4));
        
        for i=1:length(modVal)
            if ~logical(modVal(i))
                obj.previousUpdateTimes(i)      = time.time;
                obj.previousUpdateWallTimes(i)  = time.wallTime;
            end
        end
    end
    
end%PROTECTED
            
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
            
        %_________________________________________________________________________________________ logPrint
% Prints information to the standard output as well as a log file.
    function logPrint(printLine, varargin)
        if mpi_amirank0() 
            fprintf(printLine, varargin{:});
        end
    end

%_________________________________________________________________________________________ logAllPrint
% Prints information to the standard output as well as a log file for all ranks
    function logAllPrint(printLine, varargin)
            fprintf('RANK %i: ', int32(mpi_myrank()));
            fprintf(printLine, varargin{:});
    end
    
% logMaskprint
% Prints information iff any(mpi_myrank() == masks)
    function logMaskPrint(mask, printLine, varargin)
        r = mpi_myrank();
        if any(r == mask)
            fprintf('RANK %i: ', int32(r));
            fprintf(printLine, varargin{:});
        end
    end

        
    end%STATIC
    

end%CLASS
