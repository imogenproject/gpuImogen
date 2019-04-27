classdef ImogenManager < handle
% This is the primary management class for the Imogen code. It handles storage of all of the 
% un-typed variables and stores the other, specific managers. After being initialized, this object
% is passed throughout the code as the source for dynamically accessed data. This is a singleton 
% class to be accessed using the getInstance() method and not instantiated directly.
    
%===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]
        DEFAULT  = 'def';        % ENUMERATION: "Defaulted" warning
        OVERRIDE = 'over';        % ENUMERATION: "Override" warning
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public, Transient = true) %         P U B L I C  [P]
        about;          % Information about run (includes run code).                    str
        version;        % Imogen version for the run.                                   double
        detailedVersion;% Version of Imogen for the run including detailed information. str
        matlab;         % Information regarding matlab version being used.              struct
        info;           % Information generated during initialization and run.          cell(?,2)
        iniInfo;        % Information regarding data initial condition settings.        str
        DEBUG;          % Specifies if Imogen is run in debug mode.                     logical    
        defaultGamma    % Polytropic index for the run.                                 double
        PROFILE;        % Specifies state of profiler for the run.                      logical
        paths;          % Contains various paths needed for saving data.                Paths
        fades;          %

        pureHydro;      % if true, stores no magnetic information and uses simpler flux kernels

        cfdMethod;
        
        peripheralListRoot;
        eventListRoot;

        %--- Manager Classes ---%
        bc;             % Manages boundary conditions.                              BCManager

        %--- Core operations: Timestep control, conservative CFD/MHD
        time;           % Manages temporal actions.                                 TimeManager
        fluid;          % Manages fluid routines.                                   FluidManager
        radiation;      % Radiative emission                                        Radiation
        magnet;         % Manages magnetic routines.                                MagnetManager
        geometry;       % Parallel behavior/semantics/global geometry handler       GeometryManager

        %--- Source/Sink or nonideal behavior control
        selfGravity;    % Manages dynamic self-gravity solver.                      GravityManager
        potentialField; % Manages a scalar potential                                PotentialFieldManager
        
        multifluidDragMethod; % 0 = explicit midpt, 1 = RK4, 2 = exponential midpt  integer
        compositeSrcOrders;

        VTOSettings; % 

        checkpointInterval;

        %--- Saving/output
        image;          % Manages image generation and saving.                      ImageManager
        save;           % Manages updating and saving data.                         SaveManager
    end%PUBLIC
    
%===================================================================================================
    properties (SetAccess = private, GetAccess = private, Transient = true) %    P R I V A T E   [P]
        infoIndex;      % Stores current index for the info cell array.             double
        pAbortTime;     % Time of last abort check.                                 serial date #
        pAbortFile;     % Name of the abort check file.                             str
        pWarnings;      % Warning information generated during ini and run.         str
        pNotes;         % Detailed user generated information about the run.        str       
    end%PRIVATE
    
%===================================================================================================
    properties (Dependent = true, SetAccess = public) %                        D E P E N D E N T [P]
        warnings;       % Accesses formatted warning statements logged during run.      str
        notes;          % String of information regarding all manner of run activity.   str
    end %DEPENDENT
    
%===================================================================================================
    methods %                                                                    G E T / S E T   [M]    
        
%_________________________________________________________________________________________ GS: notes
% The "notes" variable. On get, this variable is formatted for wiki syntax output.
        function result = get.notes(self)
            if isempty(self.pNotes)
                result = '\n   * ENTER INFORMATION ABOUT THIS RUN HERE.';
            else
                result = strrep(self.pNotes,  '\n', '\n   * ' );
                result = strrep(result, newline(), '\n   * ' );
                result = sprintf('\n   * %s',result);
            end
        end
        
        function set.notes(self,value); self.pNotes = value; end
        
%______________________________________________________________________________________ GS: warnings
% Returns the formatted warnings statements in wiki syntax
        function result = get.warnings(self)
            if isempty(self.pWarnings)
                result = '\n   * No warnings logged.';
            else
                result = strrep(self.pWarnings,  '\n', '\n   * ' );
                result = strrep(result, newline(), '\n   * ' );
            end     
        end
        
%______________________________________________________________________________________ G: infoIndex
        function value = get.infoIndex(self); value = self.infoIndex; end
        
    end%GET/SET    
    
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]

        %_____________________________________________________________________________ ImogenManager
        function self = ImogenManager(initStruct)
        % Creates a new ImogenManager object and initializes default values.
            if mpi_any(1.0*nargin < 1)
                mpi_error('ImogenManager must be passed an initializer structure (see ''doc Initializer'')'); 
            end

            % Serve as roots for the event queuing system lists
            self.peripheralListRoot = LinkedListNode();
            self.peripheralListRoot.whatami = 'ImogenManager.peripheralListRoot';
            self.eventListRoot = LinkedListNode();
            self.eventListRoot.whatami = 'ImogenManager.eventListRoot';
            
            self.bc          = BCManager();                  self.bc.parent           = self;
            self.time        = TimeManager();                self.time.parent         = self;
            self.save        = SaveManager();                self.save.parent         = self;
            self.image       = ImageManager();               self.image.parent        = self;
            self.radiation   = Radiation();

            self.potentialField = PotentialFieldManager();   self.potentialField.parent = self;
            self.selfGravity = GravityManager();             self.selfGravity.parent  = self;
            % uploadDataArrays() handles creating these
            % self.fluid      = FluidManager();               self.fluid.parent        = self;
            self.magnet      = MagnetManager();              self.magnet.parent       = self;

            self.attachPeripheral(self.save); % temporary?

            self.paths       = Paths();
            self.info        = cell(30,2);
            self.infoIndex   = 1;
            self.DEBUG       = false;
            self.PROFILE     = false;
            self.pAbortTime  = rem(now,1);
            self.pAbortFile  = '/state.itf';
            self.matlab      = ver('matlab');
            self.pureHydro   = 0;
            self.cfdMethod   = ENUM.CFD_HLLC;

            self.VTOSettings = 0; % disables unless enabled in initialize.m
            self.checkpointInterval = 0;  % disables unless enabled
            
            self.parseIni(initStruct);
        end

        function setNumFluids(self, N)
        % FIXME check if anything exists before blindly nuking it
            self.fluid = FluidManager.empty(N, 0);
        end
        
        function parseIni(self, ini)
            
            nProblems = [0 0]; % [errors, warnings];
            
            fclose all; % Prevent any lingering saves from disrupting run.

            [self.version, self.detailedVersion] = versionInfo();
            [self.paths.hostName, self.paths.imogen, self.paths.results] = determineHostVariables();
            
            self.geometry = GeometryManager(ini.geometry);

            self.setNumFluids(ini.numFluids);
            
            if ini.numFluids > 1
                m = ini.multifluidDragMethod;
                if (m < 0) || (m > 5)
                    warning(['initializer had invalid multifluid method ' num2str(m) '. Defaulting to logtrap 2.']);
                    nProblems(2) = nProblems(2)+1;
                    m = ENUM.MULTIFLUID_LOGTRAP2;
                end
                
                self.multifluidDragMethod = m;
                
                fmnames = {'explicit midpt', 'classic rk4', 'ETD-RK1', 'ETD-RK2 (not impl)', 'LogTrap2', 'LogTrap3'};
                self.save.logPrint(['    Multifluid mode is active: multifluid drag method is ' fmnames{m+1} '\n']);
            end

            cso = ini.compositeSourceOrders;
            
            if (any(cso(1) == [2 4]) & any(cso(2) == [2 4 6])) == 0
                cso = [2 4];
                warning('Input values to composite sourcer invalid. Defaulting to 2nd order space, 4th order time.');
                nProblems(2) = nProblems(2)+1;
            end
            self.compositeSrcOrders = cso;

            if ini.numFluids > 1
                SaveManager.logPrint(['    cudaSourceComposite will have space order ' num2str(cso(1)) ' and time order ' num2str(cso(2)) '.\n']);
            end

            if ~isempty(ini.checkpointSteps)
                self.checkpointInterval = ini.checkpointSteps(1);
            else
                self.save.logPrint(['    CHECKPOINTING ENABLED with interval of ' num2str(self.checkpointInterval) 'steps.\n']);
            end
            
            if (ini.pureHydro == true) || (ini.pureHydro == 1)
                self.pureHydro = 1;
            else
                self.pureHydro = 0;
            end
            
            % Put any peripherals we're given onto the manager's peripheral list
            if isfield(ini, 'peripherals')
                self.attachPeripheral(ini.peripherals);
            end

            % Process radiation settings, if any
            if ~isempty(ini.radiation)
                self.radiation.readSubInitializer(ini.radiation);
                SaveManager.logPrint('    Radiation enabled.');
            end
            
            % Process VTO (Vacuum Taffy Operator) settings (unphysically forces a quiescent
            % near-vacuum)
            if ~isempty(ini.VTOSettings)
                self.VTOSettings = [1 ini.VTOSettings];
                self.save.logPrint(['    Vacuum Taffy Operator enabled: ' mat2str(ini.VTOSettings) '\n']);
            end

            % Process the boundary conditions structure
            if isa(ini.bcMode, 'struct')
                self.bc.modes = ini.bcMode;
                % FIXME: check for a complex-boundary-condition flag here!!!!
            elseif isa(ini.bcMode, 'char')
                modes.x = ini.bcMode; modes.y = ini.bcMode; modes.z = ini.bcMode;
                self.bc.modes = modes;
            else
                warning(['BoundaryConditionError: Boundary condition field of type %s is not recognized.' ...
                    ' bcMode recognizes string or structure input. Run aborted.'],class(ini.bcMode));
                nProblems(1) = nProblems(1)+1;
            end
            
            % Determine if the run is being profiled
            if isfield(ini, 'profile')
                self.PROFILE = ini.profile;
                if self.PROFILE
                    warning('NOTICE: Profiler will be active for this run!!!');
                    nProblems(2) = nProblems(2)+1;
                end
            end
            
            nProblems = nProblems + self.time.parseIni(ini);
            
            self.paths.runCode = ini.runCode;
            self.paths.alias   = ini.alias;
            self.about         = ini.info; % ??
            self.notes         = ini.notes;
            self.iniInfo       = ini.iniInfo; % ??
            
            self.defaultGamma = ini.gamma; % 
            
            % Parse save manager properties
            self.save.parseIni(ini);
            
            % Parse image manager properties
            self.image.parseIni(ini);

            % Ancient crap?
            self.addFades(ini.fades);
            
            % errors?
            if any(mpi_max(nProblems(1)))
                self.save.logPrint(['==========\nFATAL: One or more ranks reported ' mat2str(nProblems(1)) ' errors: Cannot continue, aborting.\n==========\n']);
                mpi_errortest(nProblems(1));
            end
            if any(mpi_max(nProblems(2)))
                self.save.logPrint('==========\nWARNING: Problems were reported by initializers\nWARNING: This run may fail; Fix problems!\n==========\n');
            end
        end
        
%_______________________________________________________________________________________ initialize
% Run pre-simulation initialization actions that require already initialized initial conditions for
% the primary array objects.
        function initialize(self, IC, mag)
            p = self.peripheralListRoot.Next;

            f = self.fluid; % fetch the fluid manager
      
            while ~isempty(p)
                p.initialize(IC, self, self.fluid, mag);
                p = p.Next;
            end

% FIXME: All these should take just 'f' as their fluid state arg
% FIXME: All these should be peripherals subsumed under the above loop, not bespoke "special" things Imogen does.
            self.image.initialize();
            for n = 1:numel(f)
                f(n).initialize(mag);
            end
            self.radiation.initialize(self, self.fluid, self.magnet);
            self.selfGravity.initialize(IC.selfGravity, f(1).mass);
            self.potentialField.initialize(IC.potentialField);
            
            if self.potentialField.ACTIVE == 0
                self.compositeSrcOrders(1) = 0;
                % disable potential field deriv calculation if not used
            end
            
            self.geometry.frameRotationCenter = IC.ini.frameParameters.rotateCenter;
            self.geometry.frameRotationOmega  = IC.ini.frameParameters.omega;
            
            if self.geometry.frameRotationOmega ~= 0
                j = self.geometry.frameRotationOmega;
                self.geometry.frameRotationOmega = 0; % alterFrameRotation reads/adjusts this, reset per fluid
                source_alterFrameRotation(self, f, j);
                
                % If the frame was boosted, we need to rebuild any statics in use
                % FIXME this should check if static BCs are actually in use
                %for n=1:numel(f)
                %    f(n).mom(1).setupBoundaries(); 
                %    f(n).mom(2).setupBoundaries();
                %    f(n).mom(3).setupBoundaries();
                %    f(n).ener.setupBoundaries();
                %end
            end
            
            % Now that everything is setup, invoke boundary condition setter for 1st time:
            for n = 1:numel(self.fluid)
                self.fluid(n).setBoundaries(1);
                self.fluid(n).setBoundaries(2);
                self.fluid(n).setBoundaries(3);
            end
            
        end
        
%_____________________________________________________________________________________ postliminary
% Function to be called at the end of a run to deactivate and cleanup remaining resources before
% ending the run entirely.
        function finalize(self, fluids, mag)
            
            %--- Copy the log file to the results directory ---%
            logFile = evalin('base','logFile;');
            if ~isempty(logFile)
                copyfile(logFile,[self.paths.save '/logfile.out']);
            end
            
            %--- Stop and save code profiling if active ---%
            if self.PROFILE
                profile('off');
                proInfo = profile('info'); %#ok<NASGU>
                save(strcat(self.paths.save, filesep, 'profile'),'proInfo'); %#ok<CPROPLC>
            end

            %--- call finalize functions of all peripherals ---%
            p = self.peripheralListRoot.Next;
            while ~isempty(p)
                p.finalize(self, fluids, mag);
                p = p.Next;
            end

            gm = GPUManager.getInstance();
            GPU_ctrl('destroyStreams',gm.deviceList,fluids(1).mass.streamptr);
        end
    
%__________________________________________________________________________________________ addFades
        function addFades(self,iniFades)
            if isempty(iniFades); return; end
            
            ids = {ENUM.MASS, ENUM.MOM, ENUM.ENER, ENUM.MAG, ENUM.GRAV};
            self.fades = cell(1,length(iniFades));
            % FIXME: wat the hecky heck is this doing?
            %for i=1:length(iniFades)
            %    switch iniFades.type
            %        case ENUM.POINT_FADE;      
            %            self.fades{i} = PointFade(self.gridSize, iniFades.location, iniFades.size);
            %    end
            %    
            %    self.fades{i}.fluxes     = iniFades.fluxes;
            %    self.fades{i}.activeList = iniFades.active;
            %    for n=1:length(iniFades.active)
            %        if ~any(strcmp(iniFades.active{n},ids))
            %            warning('ImogenManager:Fade', 'Unable to resolve fade for %s.', ...
            %                        iniFades.active{n});
            %        end
            %    end
            %end
        end
        
        function yn = chkpointThisIter(self)
            yn = self.checkpointInterval & mod(self.time.iteration, self.checkpointInterval) == (self.checkpointInterval-1);
        end

%________________________________________________________________________________________ appendInfo
% Appends an info string and value to the info cell array.
% * info    the information string                                                        srt
% * value    the value corresponding to the information string                            *
        function appendInfo(self, info, value)
            
            %--- Resize info cell array on overflow ---%
            if (self.infoIndex > size(self.info,1))
                self.info = [self.info; cell(30,2)];
            end
            
            %--- Append information to info cell array ---%
            self.info{self.infoIndex,1} = info;
            self.info{self.infoIndex,2} = ImogenRecord.valueToString(value);
            self.infoIndex = self.infoIndex + 1;
        end
        
%_____________________________________________________________________________________ appendWarning
% Appends a warning string (with value if necessary) to the warning string.
% * warning        the warning string                                                        str
% * (type)        the kind of warning ( >DEFAULT< or OVERRIDE)                            ENUM
% * (value)        numeric value related warning                                            *
        function appendWarning(self, warning, type, value)
            if (nargin < 3 || isempty(type) ),      type = self.DEFAULT; end
            if (nargin < 4 || isempty(value) ),     value = '';            end
            switch (type)
                case 'over',    typeStr = 'OVERRIDE';
                case 'def',     typeStr = 'DEFAULT';
            end
            newWarning = sprintf('\n%s: %s',typeStr,warning);
            self.pWarnings = [self.pWarnings, newWarning];
            if ~isempty(value), self.pWarnings = [self.pWarnings, sprintf('\t\t%s', ...
                                                ImogenRecord.valueToString(value))]; end
        end
       
        % call-once
        function attachPeripheral(self, p)
            if numel(p) == 1
                if isa(p, 'cell')
                    p = p{1};
                end
                p.insertAfter(self.peripheralListRoot);
            else
                for n = 1:numel(p)
                    p{n}.insertAfter(self.peripheralListRoot);
                end
            end
        end

        % call-rarely
        function attachEvent(self, e)
            e.insertAfter(self.eventListRoot);
        end

        % call-often
        function pollEventList(self, fluids, mag)
            p = self.eventListRoot.Next;
            
            while ~isempty(p)
                triggered = 0;
                if p.armed
                    if self.time.iteration >= p.iter; triggered = 1; p.armed = 0; end
                    if self.time.time      >= p.time; triggered = 1; p.armed = 0; end
                    if ~isempty(p.testHandle)
                        triggered = p.testHandle(p, self, fluids, mag);
                        if triggered; p.armed = 0; end
                    end
                end

                % the callback may delete p
                q = p.Next;
                if triggered
                    p.callbackHandle(p, self, fluids, mag);
                end
                p = q;
            end
        end

%________________________________________________________________________________________ abortCheck
% Determines if the state.itf file abort bit has been set, and if so adjusts the time manager so
% that the active run will complete on the next iteration.
        function abortCheck(self)
            %--- Initialization ---%
            if self.time.iteration >= self.time.ITERMAX;    return; end %Skip if already at run's end
       
        end 
    end%PUBLIC
    
%===================================================================================================
    methods (Access = private) %                                                  P R I V A T E  [M]
        
        
    end%PRIVATE
    
%===================================================================================================    
    methods (Static = true) %                                                      S T A T I C   [M]

    end%STATIC
    
    
end %CLASS
