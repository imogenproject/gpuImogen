function run = initialize(ini)
% This routine runs through all of the initialization variables testing to see if they've been
% set by user input. If not, then they are defaulted with a message to warn that a default
% value has been set. The values are parsed from the ini structure and then passed back as
% separate variables for explicit use in the code.
%
%>> ini          structure containing run related variable information              struct
%<< run          initialized results                                                ImogenManager

%% ============================= BEGIN RUNVAL PROPERTY INITIALIZATION =========================== %

%--- Clear all manager singletons from possible previous runs ---%
clear('ImogenManager','ImageManager','GravityManager', ...
    'MagnetManager', 'BCManager');

fclose all; % Prevent any lingering saves from disrupting run.

run             = ImogenManager();
[run.version, run.detailedVersion]                        = versionInfo();
[run.paths.hostName, run.paths.imogen, run.paths.results] = determineHostVariables();

run.geometry = GeometryManager(ini.geometry.globalDomainRez, ini.bcMode);
run.geometry.deserialize(ini.geometry);

run.setNumFluids(ini.numFluids);
if ini.numFluids > 1
    m = ini.multifluidDragMethod;
    if (m < 0) || (m > 4)
	warning(['initializer had invalid multifluid method ' num2str(m) '. Defaulting to explicit midpoint.']);
        m = 0;
    end

    run.multifluidDragMethod = m;
    if mpi_amirank0();
        fmnames = {'explicit midpt', 'classic rk4', 'ETD-RK1', 'ETD-RK2 (not impl)', 'logtrap'};
        disp(['    Multifluid mode is active: multifluid drag method to ' fmnames{m+1}]);
    end
end

cso = ini.compositeSourceOrders;

if (any(cso(1) == [2 4]) & any(cso(2) == [2 4 6])) == 0
    cso = [2 4];
    warning('Input values to composite sourcer invalid. Defaulting to 2nd order space, 4th order time.');
end
run.compositeSrcOrders = cso;

if mpi_amirank0()
    disp(['    If used, cudaSourceComposite will have space order ' num2str(cso(1)) ' and time order ' num2str(cso(2)) '.']);
end

if ~isempty(ini.checkpointSteps)
    run.checkpointInterval = ini.checkpointSteps(1);
else
    if mpi_amirank0();
        disp('WARNING')
        disp('No checkpoint interval given, checkpointing disabled')
    end
end

%% ===== GPU settings ===== %%
if (ini.pureHydro == true) || (ini.pureHydro == 1)
    run.pureHydro = 1;
else
    run.pureHydro = 0;
end

% Insert members of ini.peripherals{} using run.attachPeripheral().
if isfield(ini, 'peripherals')
    run.attachPeripheral(ini.peripherals);
end

%% ===== Radiation settings =====%%
if ~isempty(ini.radiation)
    run.radiation.readSubInitializer(ini.radiation);
    if mpi_amirank0()
        disp('Radiation subsystem enabled.');
    end
end

%% .VTOSettings Vaccum Taffy Operator (background quiescence enforcement) settings
if ~isempty(ini.VTOSettings)
    run.VTOSettings = [1 ini.VTOSettings];
    if mpi_amirank0()
        disp(['Vacuum Taffy Operator enabled: ' mat2str(ini.VTOSettings)]);
    end
end

%% .bcMode                      Edge condition modes
try
    if isa(ini.bcMode, 'struct')
        run.bc.modes = ini.bcMode;
        run.appendInfo('BC Mode', run.bc.modes);
    elseif isa(ini.bcMode, 'char')
        modes.x = ini.bcMode; modes.y = ini.bcMode; modes.z = ini.bcMode;
        run.bc.modes = modes;
    else
        error(['BoundaryConditionError: Boundary condition field of type %s is not recognized.' ...
            ' bcMode recognizes string or structure input. Run aborted.'],class(ini.bcMode));
    end
    run.appendInfo('Boundary Conditions', run.bc.modes);
catch MERR, loc_initializationError('bcMode',MERR);
end

%% .cfl                         CFL prefactor
try
    run.time.CFL = ini.cfl;
    run.appendInfo('CFL Prefactor', run.time.CFL);
catch MERR, loc_initializationError('cfl',MERR);
end

%% .thresholdMass               Threshold value below which gravity will not act

%HACK HACK HACK disabled for multifluid
%try
%    run.fluid.MASS_THRESHOLD = ini.thresholdMass;
%    run.appendInfo('Mass Threshold', run.fluid.MASS_THRESHOLD);
%catch MERR, loc_initializationError('thresholdmass',MERR);
%end

%% .profile                     Enable the profiler to record execution information

try
    run.PROFILE = ini.profile;
    if (run.PROFILE); run.appendWarning('MATLAB profiler will be active for this run.'); end
    run.appendInfo('Profiler', run.PROFILE);
catch MERR, loc_initializationError('profile',MERR);
end

%% .iterMax                     Maximum number of iterations

try
    run.time.ITERMAX = ceil(ini.iterMax);
    run.appendInfo('Maximum iteration',run.time.ITERMAX);
catch MERR, loc_initializationError('iterMax',MERR);
end

%% .timeMax                     Maximum simulation time

try
    run.time.TIMEMAX = ini.timeMax;
    run.appendInfo('Maximum simulation time.',run.time.TIMEMAX);
catch MERR, loc_initializationError('timeMax',MERR);
end

%% .wallMax                     Maximum wall time used for the run

try
    run.time.WALLMAX = ini.wallMax;
    run.appendInfo('Maximum allowed wall time (hours).',run.time.WALLMAX);
catch MERR, loc_initializationError('wallTimeMax',MERR);
end

%% .runCode                     Run code for the simulation

try
    run.paths.runCode = ini.runCode;
catch MERR, loc_initializationError('runCode',MERR);
end

%% .alias                       Alias for the simulation

try
    run.paths.alias = ini.alias;
catch MERR, loc_initializationError('alias',MERR);
end

%% .info                        Run information String

try
    run.about = ini.info;
catch MERR, loc_initializationError('info',MERR);
end

%% .notes                       Add notes (user generated)

try
    run.notes = ini.notes;
catch MERR, loc_initializationError('notes',MERR);
end

%% .iniInfo                     Add initialization information (procedurally generated)

try
    run.iniInfo = ini.iniInfo;
catch MERR, loc_initializationError('iniinfo',MERR);
end

%% .gamma                       Polytropic index for equation of state

try
    run.defaultGamma = ini.gamma;
    run.appendInfo('Default gamma value was ', run.defaultGamma);
catch MERR, loc_initializationError('gamma',MERR);
end

%% .debug                       Run the code in debug mode

try
    run.DEBUG = ini.debug;
    if (run.DEBUG), run.appendWarning('Running in debug mode.'); end
catch MERR, loc_initializationError('debug',MERR);
end

%% .save                        Save data to files

try
    run.save.FSAVE = logical(ini.save);
catch MERR, loc_initializationError('save',MERR);
end

%% .ppSave                      Percentage executed between saves

try
    run.save.PERSLICE(1) = ini.ppSave.dim1;
    run.save.PERSLICE(2) = ini.ppSave.dim2;
    run.save.PERSLICE(3) = ini.ppSave.dim3;
    run.save.PERSLICE(4) = ini.ppSave.cust;
catch MERR, loc_initializationError('ppSave',MERR);
end

%% .format
try
    run.save.format = ini.saveFormat;
catch MERR, loc_initializationError('format',MERR);
end

%% .slice                       Index Locations for slice and image save files

try
    run.save.SLICEINDEX = ini.slice;
    run.appendInfo('Slices will be saved at',run.save.SLICEINDEX);
catch MERR, loc_initializationError('slice',MERR);
end

%% .activeSlices                Which data slices to save

try
    slLabels = {'x','y','z','xy','xz','yz','xyz','cust'};
    for i=1:8
        if ~isfield(ini.activeSlices,slLabels{i}); run.save.ACTIVE(i) = false;
        else run.save.ACTIVE(i) = logical(ini.activeSlices.(slLabels{i}));
        end
        if run.save.ACTIVE(i)
            run.appendInfo('Saving slice', upper(slLabels{i}));
        end
    end
catch MERR, loc_initializationError('activeSlices',MERR);
end

%% .customSave                  Custom saving properties

try
    saveStr = '''slTime'',''slAbout'',''version'',''slGamma'',''sldGrid''';
    
    custom = ini.customSave;
    if isstruct(custom)
        if (isfield(custom,'mass')  && custom.mass),    saveStr = [saveStr ',''slMass''']; end
        if (isfield(custom,'mom')   && custom.mom),     saveStr = [saveStr ',''slMom'''];  end
        if (isfield(custom,'ener')  && custom.ener),    saveStr = [saveStr ',''slEner''']; end
        if (isfield(custom,'mag')   && custom.mag),     saveStr = [saveStr ',''slMag'''];  end
        run.save.customSaveStr = saveStr;
        customStr = 'Active';
    else
        run.save.customSaveStr = '';
        customStr = 'Inactive';
    end
    run.appendInfo('Custom save', customStr);
catch MERR, loc_initializationError('customSave',MERR);
end

%% .specSaves                   Specific save iterations

try
    if ~isempty(ini.specSaves) && isa(ini.specSaves,'double')
        run.save.specialSaves3D = ini.specSaves;
        run.appendInfo('Special save points 3D', run.save.specialSaves3D);
    else
        if isfield(ini.specSaves,'dim1')
            run.save.specialSaves1D = ini.specSaves.dim1;
            run.appendInfo('Special save points 1D', run.save.specialSaves1D);
        end
        
        if isfield(ini.specSaves,'dim2')
            run.save.specialSaves2D = ini.specSaves.dim2;
            run.appendInfo('Special save points 2D', run.save.specialSaves2D);
        end
        
        if isfield(ini.specSaves,'dim3')
            run.save.specialSaves3D = ini.specSaves.dim3;
            run.appendInfo('Special save points 3D', run.save.specialSaves3D);
        end
    end
catch MERR, loc_initializationError('specSaves',MERR);
end

%% .image                       Image saving properties

try
    
    fields = ImageManager.IMGTYPES;
    for i=1:length(fields)
        if isfield(ini.image,fields{i})
            run.image.(fields{i}) = ini.image.(fields{i});
        end
        if isfield(ini.image,'logarithmic') && isfield(ini.image.logarithmic,fields{i})
            run.image.logarithmic.(fields{i}) = ini.image.logarithmic.(fields{i});
        end
    end
    run.image.activate();
    
    if run.image.ACTIVE
        
        if isfield(ini.image,'interval')
            run.image.INTERVAL = max(1,ini.image.interval);
        else
            run.image.INTERVAL = 1;
            run.appendWarning('Image saving interval set to every step.');
        end
        
        if isfield(ini.image,'colordepth');    colordepth = ini.image.colordepth;
        else                                    colordepth = 256;
        end
        
        if isfield(ini.image,'colormap'); run.image.createColormap(ini.image.colormap, colordepth);
        else                                  run.image.createColormap('jet',colordepth);
        end
        
        imageSaveState = 'Active'; % FIXME: Wh... why is this a string?
    else imageSaveState = 'Inactive';
    end
    run.appendInfo('Image saving is', imageSaveState);
    
    if isfield(ini.image,'parallelUniformColors');
        run.image.parallelUniformColors = ini.image.parallelUniformColors; end
    
catch MERR, loc_initializationError('image',MERR);
end

%% .fades                       Fade objects
try
    run.addFades(ini.fades);
catch MERR, loc_initializationError('fades',MERR);
end

% fixme: this is overwritten up the state uploader...
if 0;
    for F = 1:ini.numFluids; % HACK HACK HACK
        %% .viscosity                   Artificial viscosity settings
        try
            run.fluid(F).viscosity.type                      = ini.viscosity.type;
            run.fluid(F).viscosity.linearViscousStrength     = ini.viscosity.linear;
            run.fluid(F).viscosity.quadraticViscousStrength  = ini.viscosity.quadratic;
        catch MERR, loc_initializationError('viscosity', MERR);
        end
    end
end

end

function loc_initializationError(property, caughtError)
% Handles errors thrown by the try statements in the initialization routine.
%
%>> property               the property that threw the error                        str
%>> caughtError            The error captured by a try block                        error


    fprintf('\n\n--- Unable to parse property %s. Run aborted. ---\n', property);
    rethrow(caghtError);
end

