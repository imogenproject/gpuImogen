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
        function result = get.notes(obj)
            if isempty(obj.pNotes)
                result = '\n   * ENTER INFORMATION ABOUT THIS RUN HERE.';
            else
                result = strrep(obj.pNotes,  '\n', '\n   * ' );
                result = strrep(result, newline(), '\n   * ' );
                result = sprintf('\n   * %s',result);
            end
        end
        
        function set.notes(obj,value); obj.pNotes = value; end
        
%______________________________________________________________________________________ GS: warnings
% Returns the formatted warnings statements in wiki syntax
        function result = get.warnings(obj)
            if isempty(obj.pWarnings)
                result = '\n   * No warnings logged.';
            else
                result = strrep(obj.pWarnings,  '\n', '\n   * ' );
                result = strrep(result, newline(), '\n   * ' );
            end     
        end
        
%______________________________________________________________________________________ G: infoIndex
        function value = get.infoIndex(obj); value = obj.infoIndex; end
        
    end%GET/SET    
    
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]

        %_____________________________________________________________________________ ImogenManager
        function obj = ImogenManager()
        % Creates a new ImogenManager object and initializes default values.

            % Serve as roots for the event queuing system lists
            obj.peripheralListRoot = LinkedListNode();
            obj.peripheralListRoot.whatami = 'ImogenManager.peripheralListRoot';
            obj.eventListRoot = LinkedListNode();
            obj.eventListRoot.whatami = 'ImogenManager.eventListRoot';
            
            
            obj.bc          = BCManager();                  obj.bc.parent           = obj;
            obj.time        = TimeManager();                obj.time.parent         = obj;
            obj.save        = SaveManager();                obj.save.parent         = obj;
            obj.image       = ImageManager();               obj.image.parent        = obj;
            obj.radiation   = Radiation();

            obj.potentialField = PotentialFieldManager();   obj.potentialField.parent = obj;
            obj.selfGravity = GravityManager();             obj.selfGravity.parent  = obj;
            % uploadDataArrays() builds & assigns an array of these now
            %obj.fluid      = FluidManager();               obj.fluid.parent        = obj;
            obj.magnet      = MagnetManager();              obj.magnet.parent       = obj;

            obj.attachPeripheral(obj.save); % temporary?

            obj.paths       = Paths();
            obj.info        = cell(30,2);
            obj.infoIndex   = 1;
            obj.DEBUG       = false;
            obj.PROFILE     = false;
            obj.pAbortTime  = rem(now,1);
            obj.pAbortFile  = '/state.itf';
            obj.matlab      = ver('matlab');
            obj.pureHydro   = 0;
            obj.cfdMethod   = ENUM.CFD_HLLC;

            obj.VTOSettings = 0; % disables unless enabled in initialize.m
            obj.checkpointInterval = 0;  % disables unless enabled
        end

        function setNumFluids(obj, N)
        % FIXME check if anything exists before blindly nuking it
            obj.fluid = FluidManager.empty(N, 0);
        end
        
%_______________________________________________________________________________________ initialize
% Run pre-simulation initialization actions that require already initialized initial conditions for
% the primary array objects.
        function initialize(obj, IC, mag)
            p = obj.peripheralListRoot.Next;

            f = obj.fluid; % fetch the fluid manager
      
            while ~isempty(p)
                p.initialize(IC, obj, obj.fluid, mag);
                p = p.Next;
            end

% FIXME: All these should take just 'f' as their fluid state arg
% FIXME: All these should be peripherals subsumed under the above loop, not bespoke "special" things Imogen does.
            obj.image.initialize();
            for n = 1:numel(f)
                f(n).initialize(mag);
            end
            obj.radiation.initialize(obj, obj.fluid, obj.magnet);
            obj.selfGravity.initialize(IC.selfGravity, f(1).mass);
            obj.potentialField.initialize(IC.potentialField);
            
            if obj.potentialField.ACTIVE == 0
                obj.compositeSrcOrders(1) = 0;
                % disable potential field deriv calculation if not used
            end
            
            obj.geometry.frameRotationCenter = IC.ini.frameParameters.rotateCenter;
            obj.geometry.frameRotationOmega  = IC.ini.frameParameters.omega;
            
            if obj.geometry.frameRotationOmega ~= 0
                j = obj.geometry.frameRotationOmega;
                obj.geometry.frameRotationOmega = 0; % alterFrameRotation reads/adjusts this, reset per fluid
                source_alterFrameRotation(obj, f, j);
                
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
            for n = 1:numel(obj.fluid)
                obj.fluid(n).setBoundaries(1);
                obj.fluid(n).setBoundaries(2);
                obj.fluid(n).setBoundaries(3);
            end
            
        end
        
%_____________________________________________________________________________________ postliminary
% Function to be called at the end of a run to deactivate and cleanup remaining resources before
% ending the run entirely.
        function finalize(obj, fluids, mag)
            
            %--- Copy the log file to the results directory ---%
            try
                logFile = evalin('base','logFile;');
                if ~isempty(logFile)
                    copyfile(logFile,[obj.paths.save '/logfile.out']);
                end
            catch ME
            end
            
            %--- Stop and save code profiling if active ---%
            if obj.PROFILE
                profile('off');
                proInfo = profile('info'); %#ok<NASGU>
                save(strcat(obj.paths.save, filesep, 'profile'),'proInfo');
            end

            %--- call finalize functions of all peripherals ---%
            p = obj.peripheralListRoot.Next;
            while ~isempty(p)
                p.finalize(obj, fluids, mag);
                p = p.Next;
            end

            gm = GPUManager.getInstance();
            GPU_ctrl('destroyStreams',gm.deviceList,fluids(1).mass.streamptr);
        end
    
%__________________________________________________________________________________________ addFades
        function addFades(obj,iniFades)
            if isempty(iniFades); return; end
            
            ids = {ENUM.MASS, ENUM.MOM, ENUM.ENER, ENUM.MAG, ENUM.GRAV};
            obj.fades = cell(1,length(iniFades));
            % FIXME: wat the heck is this doing?
            %for i=1:length(iniFades)
            %    switch iniFades.type
            %        case ENUM.POINT_FADE;      
            %            obj.fades{i} = PointFade(obj.gridSize, iniFades.location, iniFades.size);
            %    end
            %    
            %    obj.fades{i}.fluxes     = iniFades.fluxes;
            %    obj.fades{i}.activeList = iniFades.active;
            %    for n=1:length(iniFades.active)
            %        if ~any(strcmp(iniFades.active{n},ids))
            %            warning('ImogenManager:Fade', 'Unable to resolve fade for %s.', ...
            %                        iniFades.active{n});
            %        end
            %    end
            %end
            
            
            
        end
        
        function yn = chkpointThisIter(obj)
            yn = obj.checkpointInterval & mod(obj.time.iteration, obj.checkpointInterval) == (obj.checkpointInterval-1);
        end

%________________________________________________________________________________________ appendInfo
% Appends an info string and value to the info cell array.
% * info    the information string                                                        srt
% * value    the value corresponding to the information string                            *
        function appendInfo(obj, info, value)
            
            %--- Resize info cell array on overflow ---%
            if (obj.infoIndex > size(obj.info,1))
                obj.info = [obj.info; cell(30,2)];
            end
            
            %--- Append information to info cell array ---%
            obj.info{obj.infoIndex,1} = info;
            obj.info{obj.infoIndex,2} = ImogenRecord.valueToString(value);
            obj.infoIndex = obj.infoIndex + 1;
        end
        
%_____________________________________________________________________________________ appendWarning
% Appends a warning string (with value if necessary) to the warning string.
% * warning        the warning string                                                        str
% * (type)        the kind of warning ( >DEFAULT< or OVERRIDE)                            ENUM
% * (value)        numeric value related warning                                            *
        function appendWarning(obj, warning, type, value)
            if (nargin < 3 || isempty(type) ),      type = obj.DEFAULT; end
            if (nargin < 4 || isempty(value) ),     value = '';            end
            switch (type)
                case 'over',    typeStr = 'OVERRIDE';
                case 'def',     typeStr = 'DEFAULT';
            end
            newWarning = sprintf('\n%s: %s',typeStr,warning);
            obj.pWarnings = [obj.pWarnings, newWarning];
            if ~isempty(value), obj.pWarnings = [obj.pWarnings, sprintf('\t\t%s', ...
                                                ImogenRecord.valueToString(value))]; end
        end
       
        % call-once
        function attachPeripheral(obj, p)
            if numel(p) == 1
                if isa(p, 'cell')
                    p = p{1};
                end
                p.insertAfter(obj.peripheralListRoot);
            else
                for n = 1:numel(p)
                    p{n}.insertAfter(obj.peripheralListRoot);
                end
            end
        end

        % call-rarely
        function attachEvent(obj, e)
            e.insertAfter(obj.eventListRoot);
        end

        % call-often
        function pollEventList(obj, fluids, mag)
            p = obj.eventListRoot.Next;
            
            while ~isempty(p)
                triggered = 0;
                if p.armed
                    if obj.time.iteration >= p.iter; triggered = 1; p.armed = 0; end
                    if obj.time.time      >= p.time; triggered = 1; p.armed = 0; end
                    if ~isempty(p.testHandle)
                        triggered = p.testHandle(p, obj, fluids, mag);
                        if triggered; p.armed = 0; end
                    end
                end

                % the callback may delete p
                q = p.Next;
                if triggered
                    p.callbackHandle(p, obj, fluids, mag);
                end
                p = q;
            end
        end

%________________________________________________________________________________________ abortCheck
% Determines if the state.itf file abort bit has been set, and if so adjusts the time manager so
% that the active run will complete on the next iteration.
        function abortCheck(obj)
            %--- Initialization ---%
            if obj.time.iteration >= obj.time.ITERMAX;    return; end %Skip if already at run's end
       
        end 
    end%PUBLIC
    
%===================================================================================================
    methods (Access = private) %                                                  P R I V A T E  [M]
        
        
    end%PRIVATE
    
%===================================================================================================    
    methods (Static = true) %                                                      S T A T I C   [M]

    end%STATIC
    
    
end %CLASS
