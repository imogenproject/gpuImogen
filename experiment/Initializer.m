classdef Initializer < handle
    % Base class for data initialization objects.
    
    %===============================================================================================
    properties (Constant = true, Transient = true) %                        C O N S T A N T [P]
    end%CONSTANT
    
    %===============================================================================================
    properties (SetAccess = public, GetAccess = public) %                      P U B L I C  [P]
        activeSlices;   % Which slices to save.                             struct
        bcMode;         % Boundary value types and settings.                struct
        cfl;            % Multiplicative coefficient for timesteps.         double
        customSave;     % Which arrays to save in custom save slices.       struct
        debug;          % Run Imogen in debug mode.                         logical *
        gravity;        % Gravity sub-initializer object containing all     GravitySubInitializer
                        %   gravity related settings.
        info;           % Short information describing run.                 string
        image;          % Which images to save.                             struct
        iterMax;        % Maximum iterations for the run.                   int
        wallMax;        % Maximum wall time allowed for the run in hours.   double
        mode;           % Specifies the portions of the code are active.    struct
        notes;          % Lengthy information regarding a run.              string
        ppSave;         % Percentage of execution between slice saves.      struct
        profile;        % Specifies enabling of performance profiling.      logical
        runCode;        % Code used to specify experiment type.             string
        alias;          % Unique identifier for the run.                    string
        save;           % Enable/disable saving data to files for a run.    logical
        saveFormat;     % Either ENUM.FORMAT_MAT, ENUM.FORMAT_NC or ENUN.FORMAT_H5 integer
        slice;          % Indices for slices when saving data (1x3).        int
        specSaves;      % Arrays of special iterations used to save data.   struct
        thresholdMass;  % Minium mass acted on by gravitational forces      double
        timeMax;        % Maximum simulation time before run exits.         double
        treadmill;      % Treadmilling direction, inactive if empty.        string
        viscosity;      % Viscosity sub initializer object.                 ViscositySubInitializer
        radiation;      % radiation sub-initializer
        logProperties;  % List of class properties to include in run.log    cell
        fluxLimiter;    % Specifies the flux limiter(s) to use.             struct

        gpuDeviceNumber;% ID of the device to attempt to run on
        pureHydro;      % if true, uses nonmagnetic flux routines

        useInSituAnalysis;%If nonzero, will do as below
        inSituHandle;   % Must be @SimulationAnalyzer.getInstance for some simulation analyzer
        stepsPerInSitu; % If > 0, Imogen will pass the sim state to the instance's FrameAnalyer() function
                        % once per this many frames
        inSituInstructions;
        peripherals;

        frameParameters;% .omega, rotateCenter, centerVelocity
        fluidDetails;
        numFluids;
        multifluidDragMethod;
        compositeSourceOrders;
        
        VTOSettings;    % [alpha beta] or blank
        
        checkpointSteps;
        
        unitValues; 
        
        geomgr;         % GeometryManager class
    end %PUBLIC

%===================================================================================================
    properties (Dependent = true) %                                       D E P E N D E N T [P]
        fades;
        saveSlicesSpecified; % Specifies if save slices have been specified.
        gamma; % Compat: feeds through to .fluidDetails(1).gamma
        minMass; % Compat: feeds through to .fluidDetails(1).minMass
    end %DEPENDENT
        
%===================================================================================================
    properties (SetAccess = private, GetAccess = private) %                   P R I V A T E [P]
        pInfo;
        pInfoIndex;
        pFades;
        pFadesIndex;
        pLoadFile;          % Specifies the file name, including full path, to      string
                            %   the loaded data object.
        pFileData;          % Structure of data arrays used to store data that      struct
                            %   was loaded from a file instead of generated by
                            %   the initializer class.
    end %PROTECTED
        
%===================================================================================================
    methods %                                                                G E T / S E T  [M]

%__________________________________________________________________________________ Initializer
        function obj = Initializer()
            obj.pInfo                = cell(1,100);
            obj.pInfoIndex           = 1;
            obj.pFadesIndex          = 1;
            obj.mode.fluid           = true;
            obj.mode.magnet          = false;
            obj.mode.gravity         = false;
            obj.debug                = false;
            obj.iterMax              = 10;
            obj.ppSave.dim1          = 10;
            obj.ppSave.dim2          = 25;
            obj.ppSave.dim3          = 50;
            obj.ppSave.cust          = 50;
            obj.profile              = false;
            obj.save                 = true;
            obj.saveFormat           = ENUM.FORMAT_HDF;
            obj.thresholdMass        = 0;
            obj.timeMax              = 1e5;
            obj.wallMax              = 1e5;
            obj.treadmill            = 0;
            obj.gravity              = SelfGravityInitializer();
            obj.fluxLimiter          = struct();

            obj.useInSituAnalysis    = 0;
            obj.stepsPerInSitu       = 99999999999;

            obj.gpuDeviceNumber            = 0;
            obj.pureHydro = 0;

            obj.frameParameters.omega = 0;
            obj.frameParameters.rotateCenter = [0 0];
            obj.frameParameters.velocity = [0 0 0];

            obj.numFluids = 1;
            obj.fluidDetails = fluidDetailModel();
            obj.fluidDetails(1) = fluidDetailModel('cold_molecular_hydrogen');

            obj.multifluidDragMethod = ENUM.MULTIFLUID_LOGTRAP2;
            obj.compositeSourceOrders = [2 4];
            
            obj.unitValues = [1 1 1]; % Units we want for mass in kg, length in meters, time in seconds

            fields = SaveManager.SLICEFIELDS;
            for i=1:length(fields)
                obj.activeSlices.(fields{i}) = false; 
            end
            
            obj.logProperties       = {'alias'};
            
            obj.geomgr = GeometryManager([128 128 128]);
        end           

%______________________________________________________________________ GS: saveSlicesSpecified
        function result = get.saveSlicesSpecified(obj)
            s      = obj.activeSlices;
            result = s.x || s.y || s.z || s.xy || s.xz || s.yz || s.xyz;
        end


        function g = get.gamma(self); g = self.fluidDetails(1).gamma; end
        function set.gamma(self, g); self.fluidDetails(1).gamma = g; end
        function m = get.minMass(self); m = self.fluidDetails(1).minMass; end
        function set.minMass(self, m); self.fluidDetails(1).minMass = m; end

        
%____________________________________________________________________________________ GS: fades
        function result = get.fades(obj)
            if (obj.pFadesIndex < 2);  result = [];
            else;                     result = obj.pFades(1:(obj.pFadesIndex-1));
            end
        end
        
%_____________________________________________________________________________________ GS: info
        function result = get.info(obj)
            if isempty(obj.info); obj.info = 'Unspecified trial information.'; end
            result = ['---+ (' obj.runCode ') ' obj.info];
        end

%____________________________________________________________________________________ GS: image
        function result = get.image(obj)
            fields = ImageManager.IMGTYPES;
            if ~isempty(obj.image);     result = obj.image; end
            for i=1:length(fields)
               if isfield(obj.image,fields{i}); result.(fields{i}) = obj.image.(fields{i});
               else; result.(fields{i}) = false;
               end
            end
            
            
            
        end
                
%______________________________________________________________________________________ GS: cfl        
        function result = get.cfl(obj)
           if isempty(obj.cfl)
               if obj.mode.magnet;      result = 0.35;
               else;                     result = 0.85;
               end
           else; result = obj.cfl;
           end
        end
        
%___________________________________________________________________________________ GS: bcMode     
        function result = get.bcMode(obj)
           if isempty(obj.bcMode)
                result.x = 'circ'; result.y = 'circ'; result.z = 'circ';
           else; result = obj.bcMode;
           end
        end
        
%______________________________________________________________________________ GS: fluxLimiter
        function result = get.fluxLimiter(obj)
            result = struct();
            fields = {'x', 'y', 'z'};
            for i=1:3
                if isfield(obj.fluxLimiter, fields{i})
                    result.(fields{i}) = obj.fluxLimiter.(fields{i});
                else
                    result.(fields{i}) = FluxLimiterEnum.VAN_LEER;
                end
            end
        end

        end%GET/SET
        
%===================================================================================================
    methods (Access = public) %                                                P U B L I C  [M]
        
%_______________________________________________________________________________ operateOnInput
        function operateOnInput(obj, input, defaultGrid)
            if isempty(input)
                grid        = defaultGrid;
                
            elseif isnumeric(input)
                grid        = input;
               
            elseif ischar(input) % FIXME this is broken for almost sure at present
                obj.pLoadFile   = input;
                obj.loadDataFromFile(input);
            end
            
            obj.geomgr.setup(grid);
        end
        
%_________________________________________________________________________ getInitialConditions
% Given simulation parameters filled out in a superclass, uses the superclass' calculateInitialConditions
% function to get Q(x,0) fluid fields.
        function [fluids, mag, statics, potentialField, selfGravity, iniSettings] = getInitialConditions(obj)
            if ~isempty(obj.pLoadFile)
                mass    = obj.pFileData.mass;
                mom     = obj.pFileData.mom;
                ener    = obj.pFileData.ener;
                mag     = obj.pFileData.mag;
                statics = StaticsInitializer();
                potentialField = PotentialFieldInitializer();
                selfGravity = SelfGravityInitializer();
                % FIXME: This will not work without recovering the fluid details structure!!! ... somehow.
                fluids = struct('mass',mass,'momX',squish(mom(1,:,:,:)),'momY',squish(mom(2,:,:,:)),'momZ',squish(mom(3,:,:,:)),'ener',ener,'details',[]);

                ini  = load([path filesep 'ini_settings.mat']);
                obj.populateValues(ini.ini);

            else
                obj.geomgr.setup(obj.geomgr.globalDomainRez, obj.bcMode);

                SaveManager.logPrint('---------- Calculating initial conditions\n');
                [fluids, mag, statics, potentialField, selfGravity] = obj.calculateInitialConditions();

                for z = 1:numel(fluids)
                    fluids(z).details = obj.fluidDetails(z);
                    fluids(z).details.minMass = mpi_max(fluids(z).details.minMass);
                end

            end

% FIXME this is an ugly hack; slice determination is a FAIL since parallelization.
%            if isempty(obj.slice)
                obj.slice = ceil(obj.geomgr.globalDomainRez/2);
%            end

            obj.activateComplexBoundaryConditions(numel(fluids));
            % If not /actually/ using them, calling it at this point just stamps out copies of the
            % bcMode structure into cells so we don't have to do any conditional junk downstream

            obj.checkForInvalidBCs();
            
            iniSettings = obj.getRunSettings();

        end

        function saveParallelInitialConditions(obj, nprocs, basename, bytime)
            
            % tuple to rank
            ttr = @(c) c(1) + nprocs(1)*(c(2)+nprocs(2)*c(3));

            obj.geomgr.topology.nproc = nprocs;
            obj.geomgr.context.size = prod(nprocs);
            
            for i = 1:nprocs(1)
                for j = 1:nprocs(2)
                    for k = 1:nprocs(3)
                        crd = [i j k] - 1;
                        obj.geomgr.topology.coord = crd;

                        myleft  = mod(crd - 1 + nprocs, nprocs);
                        myright = mod(crd + 1,         nprocs);

                        myleft(1) = ttr([myleft(1) crd(2) crd(3)]);
                        myleft(2) = ttr([crd(1) myleft(2) crd(3)]);
                        myleft(3) = ttr([crd(1) crd(2) myleft(3)]);

                        myright(1) = ttr([myright(1) crd(2) crd(3)]);
                        myright(2) = ttr([crd(1) myright(2) crd(3)]);
                        myright(3) = ttr([crd(1) crd(2) myright(3)]);

                        myrank = ttr(crd);
                        
                        obj.geomgr.context.rank = myrank;
                        obj.geomgr.topology.neighbor_left = myleft;
                        obj.geomgr.topology.neighbor_right= myright;

                        eyesee = obj.saveInitialCondsToStructure();
                        f = translateICStructToFrame(eyesee);
                        if myrank == 0 % first go, write out the initializer too
                            % Certain additions are required here:
                            translateInitializerToH5(eyesee.ini, [basename '.h5'], basename, bytime);
                            h5writeatt([basename '.h5'], '/', 'gravConstant', eyesee.potentialField.constant*(numel(eyesee.potentialField.field) > 0));
                        end
                        
                        p = ceil(log10(eyesee.ini.iterMax));
                        util_Frame2HDF(sprintf('%s_3D_XYZ_rank%03i_%0*i.h5', basename, myrank, int32(p), 0), f);

                        if numel(eyesee.potentialField.field) > 0
                            gravfile = sprintf('%s_staticpot_rank%03i.h5', basename, myrank);
                            h5create(gravfile, '/staticGravpot', size(eyesee.potentialField.field), 'Datatype', 'double');
                            h5write(gravfile, '/staticGravpot', eyesee.potentialField.field);
                        end
                        clear eyesee;
                    end
                end
            end
        end

        % These either dump ICs to a file or return them as a structure.
        function icfile = saveInitialCondsToFile(obj)
            IC = obj.saveInitialCondsToStructure();

            icfile = [tempname '.mat'];
            save(icfile, 'IC','-v7.3');
        end

        function IC = saveInitialCondsToStructure(obj)
            [fluids, mag, statics, potentialField, selfGravity, ini] = obj.getInitialConditions();
            IC.fluids = fluids;
            IC.magX = squish(mag(1,:,:,:));
            IC.magY = squish(mag(2,:,:,:));
            IC.magZ = squish(mag(3,:,:,:));
            if isempty(statics); IC.statics = StaticsInitializer(obj.geomgr); else; IC.statics = statics; end
            if isempty(potentialField); IC.potentialField = PotentialFieldInitializer(); else; IC.potentialField = potentialField; end
            if isempty(selfGravity); IC.selfGravity = SelfGravityInitializer(); else; IC.selfGravity = selfGravity; end
            IC.ini = ini;
            IC.ini.geometry = obj.geomgr.serialize(); % FIXME: this ought to perhaps happen elsewhere?
        end

        function y = locatePeripheral(obj, name)
            y = [];
             for N = 1:numel(obj.peripherals)
                 if isa(obj.peripherals{N}, name); y = obj.peripherals{N}; break; end
             end
        end
        
        function activateComplexBoundaryConditions(self, N)
        % Multifluid BCs: It we are using multiple fluids but not using complex BCs,
        % stamp out copies of the .bcMode structure into a {cell array}
            if isa(self.bcMode, 'cell') == 0
                q = cell([N 1]);
                for j = 1:N; q{j} = self.bcMode; end
                self.bcMode = q;
            end
            
        end
        
        function checkForInvalidBCs(self)
            % FIXME - expand the BC structure and check if anything glaringly not OK is done.
            % FIXME - that is primarily to say, check that all size-one dimensions are set to circ
            % because other conditions often assume the ability to read a boundary at least N cells
            % from the edge, which is not the case if size in a dimension is one!

            if isa(self.bcMode, 'cell') == 0
                bc = {self.bcMode};
            else
                bc = self.bcMode;
            end
            
            for j = 1:numel(bc)
                m = BCManager.expandBCStruct(bc{j});
                s0 = ['There is only one cell in the Y direction but the boundary mode is "%s"\nThis is an error in the ' class(self) ' initializer''s settings, it must be set to ENUM.BCMODE_CIRCULAR.\nThe problem is not fixed here because of possible further self consistency problems within the Initializer.\nViolations of this requirement cause invalid address exceptions at simulation start time.'];
                
                % trawl the Y part: if ny == 1 anything but circular is an error
                if self.geomgr.globalDomainRez(2) == 1
                    if strcmp(m{1,2}, ENUM.BCMODE_CIRCULAR) ~= 1
                        error(s0, m{1,2});
                    end
                    if strcmp(m{2,2}, ENUM.BCMODE_CIRCULAR) ~= 1
                        error(s0, m{2,2});
                    end
                end
                
                s0 = ['There is only one cell in the Z direction but the boundary mode is "%s"\nThis is an error in the ' class(self) ' initializer''s settings, it must be set to ENUM.BCMODE_CIRCULAR.\nThe problem is not fixed here because of possible further self consistency problems within the Initializer.\nViolations of this requirement cause invalid address exceptions at simulation start time.'];
                % trawl the Z part: if nz == 1 anything but circular is an error
                if self.geomgr.globalDomainRez(3) == 1
                    if strcmp(m{1,3}, ENUM.BCMODE_CIRCULAR) ~= 1
                        error(s0, m{1,3});
                    end
                    if strcmp(m{2,3}, ENUM.BCMODE_CIRCULAR) ~= 1
                        error(s0, m{2,3});
                    end
                end
            end
            
        end

%_______________________________________________________________________________ getRunSettings
        function result = getRunSettings(obj)
            % This function dumps the initializer class' members into a big fat dynamic structure for
            % initializer.m to parse

            %--- Populate skips cell array ---%
            %       Specific fields are skipped from being included in the initialization structure.
            %       This includes any field that is named in all CAPITAL LETTERS.
            fields = fieldnames(obj);
            nFields = length(fields);
            
            skips  = cell(nFields, 1);
            for i=1:nFields
                if (strcmp(upper(fields{i}),fields{i}))
                    skips{length(skips) + 1} = fields{i};
                end
            end
            
            result          = Initializer.parseValues(obj, skips);
            result.iniInfo  = obj.getInfo();
        end

        
%______________________________________________________________________________________ addFade
% Adds a fade object to the run.
        function addFade(obj, location, fadeSize, fadeType, fadeFluxes, activeList)
            index                       = obj.pFadesIndex;
            obj.pFades(index).location  = location;
            obj.pFades(index).size      = fadeSize;
            obj.pFades(index).type      = fadeType;
            obj.pFades(index).active    = activeList;
            obj.pFades(index).fluxes    = fadeFluxes;
            obj.pFadesIndex = index + 1;
        end

        end%PUBLIC
        
%===================================================================================================      
        methods (Access = protected) %                                 P R O T E C T E D    [M]
   

        
%___________________________________________________________________ calculateInitialConditions
% Calculates all of the initial conditions for the run and returns a simplified structure containing
% all of the initialization property settings for the run.

        function [fluids, mag, statics, potentialField, selfGravity] = calculateInitialConditions(obj)
            %%%% Must be implemented in subclasses %%%%
        end

%_____________________________________________________________________________ loadDataFromFile
        function loadDataFromFile(obj, filePathToLoad)
            if isempty(filePathToLoad); return; end
            % filePathToLoad should contain the prefix only, i.e. '~/Results/Dec12/blah/3D_XYZ

            % Evaluate which of the savefiles is associated with my rank.
            aboot = savefileInfo(filePathToLoad);
            myfile = sprintf('%s/%srank%i_%.*i.%s',aboot.path,aboot.prefix,aboot.rank,aboot.pad,aboot.frameno,aboot.extension);
           
            path                        = fileparts(filePathToLoad); % path to directory with savedata
            data                        = load(myfile);
            
            fields                      = fieldnames(data);
            obj.pFileData.mass          = data.(fields{1}).mass;
            obj.pFileData.ener          = data.(fields{1}).ener;
            obj.pFileData.mom           = zeros([3, size(obj.pFileData.mass)]);
            obj.pFileData.mom(1,:,:,:)  = data.(fields{1}).momX;
            obj.pFileData.mom(2,:,:,:)  = data.(fields{1}).momY;
            obj.pFileData.mom(3,:,:,:)  = data.(fields{1}).momZ;
            obj.pFileData.mag           = zeros([3, size(obj.pFileData.mass)]);
            if ~isempty(data.(fields{1}).magX)
                obj.pFileData.mag(1,:,:,:)  = data.(fields{1}).magX;
                obj.pFileData.mag(2,:,:,:)  = data.(fields{1}).magY;
                obj.pFileData.mag(3,:,:,:)  = data.(fields{1}).magZ;
            end
            
            clear('data');
            
            ini  = load([path filesep 'ini_settings.mat']);
            obj.populateValues(ini.ini);
        end
        
        
%______________________________________________________________________________________ getInfo
% Gets the information string
        function result = getInfo(obj)
            
            %--- Populate skips cell array ---%
            %       Specific fields are skipped from being included in the initialization structure.
            %       This includes any field that is named in all CAPITAL LETTERS.
            fields = fieldnames(obj);
            skips  = {'info', 'runCode'};
            for i=1:length(fields)
                if (strcmp(upper(fields{i}),fields{i}))
                    skips{length(skips) + 1} = fields{i};
                end
            end
            
            result = '';
            for i=1:length(obj.pInfo)
                if isempty(obj.pInfo{i}); break; end
                result = strcat(result,sprintf('\n   * %s',obj.pInfo{i}));
            end
            
           result = [result '\n   * Intialized settings:' ...
                    ImogenRecord.valueToString(obj, skips, 1)];
        end                
                
%___________________________________________________________________________________ appendInfo
% Adds argument string to the info string list for inclusion in the iniInfo property. A good way
% to store additional information about a run.
        function appendInfo(obj, infoStr, varargin)
            if ~isempty(varargin)
                evalStr = ['sprintf(''' infoStr ''''];
                for i=1:length(varargin)
                   evalStr = strcat(evalStr, ',varargin{', num2str(i), '}');
                end
                evalStr = strcat(evalStr,');');
                infoStr = eval(evalStr);
            end
            obj.pInfo{obj.pInfoIndex} = infoStr;
            obj.pInfoIndex = obj.pInfoIndex + 1;
        end
        
        %_______________________________________________________________________ populateValues
        function populateValues(obj, loadedInput, inputFields)
            if (nargin < 3 || isempty(inputFields)); inputFields = {}; end
            fields  = fieldnames(getfield(loadedInput, {1}, inputFields{:}, {1}));
            values  = getfield(loadedInput, {1}, inputFields{:}, {1});
            inLen   = length(inputFields);
            
            for i=1:length(fields)
                newInputFields              = inputFields;
                newInputFields{inLen + 1}   = fields{i};
                if  strcmp('fades', fields{i})
                    obj.pFades      = values.fades;
                    obj.pFadesIndex = length(obj.pFades) + 1;
                elseif  isstruct(values.(fields{i}))
                    obj.populateValues(loadedInput, newInputFields);
                else
                    objFieldStr = 'obj';
                    for j=1:length(newInputFields)
                        objFieldStr = strcat(objFieldStr, '.', newInputFields{j});
                    end
                    
                    try
                        eval([objFieldStr, ' = values.(fields{i});']);
                    catch MERR
                        if strcmp(upper(fields{i}), fields{i}); continue; end
                        if strcmp(fields{i}, 'iniInfo'); continue; end
                        
                        if strcmp(fields{i}, 'fades')
                            obj.pFades          = values.(fields{i});
                            obj.pFadesIndex     = length(obj.pFades);
                            continue;
                        end
                        
                        fprintf('\tWARNING: Unable to set value for "%s"\n',objFieldStr);
                    end
                end
            end
        end
        
        end%PROTECTED
        
        %===========================================================================================
        methods (Static = true) %                                            S T A T I C    [M]
            
            %______________________________________________________________________ parseValues
            % Parses an object and returns a structure of corresponding fields and value pairs.
            function result = parseValues(objectToParse, skips)
                fields = fieldnames(objectToParse);
                for i=1:length(fields)
                    if any(strcmp(fields{i}, skips)); continue; end
                    if isobject(objectToParse.(fields{i}))
                        result.(fields{i}) = Initializer.parseValues(objectToParse.(fields{i}), skips);
                    else
                        result.(fields{i}) = objectToParse.(fields{i});
                    end
                end
            end
            
            %___________________________________________________________________________ make3D
            % Enforces the 3D nature of an input value, so that it has a value for each spatial component.
            %>> inputValue       value to make 3D, can be length 1,2, or 3.           double(?)
            %>> fill             value to use when one is missing in the inputValue.  double
            %<< result           converted value to have 3 spatial components         double(3)
            function result = make3D(inputValue, fill)
                if nargin < 3 || isempty(fill);                fill = min(inputValue);                end
                inLen = length(inputValue);
                
                switch inLen
                    case 1;                result = inputValue * ones(1,3);
                    case 2;                result = [inputValue fill];
                    case 3;                result = inputValue;
                end
            end
            
            function f = rhoMomEtotToFluid(mass, mom, ener)
                % This is a compat function: Convert old mass/mom/ener arrays to fluid() struct.
                f = struct('mass',mass, ...
                           'momX',squish(mom(1,:,:,:),'onlyleading'), ...
                           'momY',squish(mom(2,:,:,:),'onlyleading'), ...
                           'momZ',squish(mom(3,:,:,:),'onlyleading'), ...
                           'ener',ener,'details',[], 'bcData', []);
            end
            
            function f = rhoVelEintToFluid(mass, vel, Eint)
                Etot = .5* squish(sum(vel.*vel,1),'onlyleading').* mass + Eint;
                f = struct('mass',mass, ...
                           'momX',squish(vel(1,:,:,:),'onlyleading') .* mass, ...
                           'momY',squish(vel(2,:,:,:),'onlyleading') .* mass, ...
                           'momZ',squish(vel(3,:,:,:),'onlyleading') .* mass, ...
                           'ener',Etot,'details',[], 'bcData', []);
            end
            
        end%PROTECTED
        
end%CLASS
