function initializeResultPaths(run, IC)
% Creates the directories to store results based on the user input save settings and stores them as
% a Paths object contained within the ImogenManager instance for the run.
%
%<> run              manager object for the Imogen run                                ImogenManager
%>> IC               Initial condition structure containing path, time, and grid info

    SaveManager.logPrint('--------- Preparing for output\n');

    if isfield(IC, 'originalPathStruct')
        run.paths.deserialize(IC.originalPathStruct); 
        setupDirectories = 0;
    else
        run.paths.initialize(run);
        setupDirectories = 1;
    end
   
    mpi_barrier(); % force all units to evaluate to the same paths
    run.paths.indexPadding = length(num2str(run.time.ITERMAX));

    if mpi_amirank0() && (setupDirectories == 1)
    %-----------------------------------------------------------------------------------------------
    % Determine directory names
    %--------------------------
    SaveManager.logPrint('    Creating directories... ');
            
        %--- RESULTS Directory ---%
        MErr = locDirMaker(run, run.paths.results,'Results folder Created');
        if isa(MErr,'MException')
            error('PathCreation:Results', 'Unable to create or locate results folder. Aborting run. ');
        end

        %--- DATE CONTAINER Directory ---%
        MErr = locDirMaker(run, run.paths.container,'Results container folder created');
        if isa(MErr,'MException')
           error('PathCreation:Container', ['Unable to create or locate container folder '...
                                       'inside supplied results folder. Aborting run']);
        end

        %--- RUN Directory ---%
        MErr = locDirMaker(run, run.paths.save,'Saving run to');
        if isa(MErr,'MException')
            error('PathCreation:Save', ['Unable to create directory for run. '...
                                            'Aborting operation.']);
        end

        SaveManager.logPrint('Run path: %s\n', run.paths.save);

        %--- IMAGE Directories ---%
        if ( ~isempty(run.image.ACTIVE) )

            MErr = locDirMaker(run, run.paths.image, 'Saving images to' );
            if isa(MErr,'MException')
                error('PathCreation:Image', ['Unable to create the image directory in the '...
                                                'save folder. Aborting operation.']);
            end

            fields = ImageManager.IMGTYPES;
            for i=1:length(fields)
                    
                if run.image.(fields{i})
                    MErr = locDirMaker(run, [run.paths.image '/' fields{i}], ...
                                               [upper(fields{i}(1)) fields{i}(2:end)]);
                                           
                    if isa(MErr, 'MException')
                        error('PathCreation:Image', 'Unable to create an image subdirectory. Aborting operation.');
                    end
                end
                   
                if isfield(run.image.logarithmic,(fields{i}))
                    MErr = locDirMaker(run, [run.paths.image '/log_' fields{i}], ...
                                            ['Logarithmic ' upper(fields{i}(1)) fields{i}(2:end)]);
                                           
                    if isa(MErr, 'MException')
                         error('PathCreation:Image', ['Unable to create an image '...
                                                        'subdirectory. Aborting operation.']);
                    end
                end
                   
            end

                
        end

        %--- Save Run File ---%
        %           Searches through the stack for the file that initiated the run by finding
        %           the file higher in the stack than imogen.m. If found that file is copied to
        %           run file directory for future reference.
        stack = dbstack('-completenames');
        stackLen = length(stack);
        for i=1:stackLen
            if strcmpi(stack(i).name, 'imogen')
                runIndex = i + 1;
                if runIndex > stackLen
                    warning('Imogen:RunFile', ['Unable to determine run file used to ' ...
                             'initiate simulation. Skipping file copy to save directory.']);
                else
                    status = copyfile(stack(runIndex).file, [run.paths.save '/runfile.m']);
                    if ~status; warning('Imogen:FileIO', 'Unable to copy run file to save directory.'); end
                end
            end
        end

    SaveManager.logPrint('    Waiting for global visibility of output path by all ranks: ');

    end

    timeTaken = enforceConsistentView(run.paths.save, 600, 1);
    SaveManager.logPrint('    Results directory became globally visible in %.1f sec\n', timeTaken);
    
    switch run.save.format
	case ENUM.FORMAT_MAT; sp = '    Output format: .MAT files\n';
	case ENUM.FORMAT_NC;  sp = '    Output format: .NC NetCDF files\n';
	case ENUM.FORMAT_HDF; sp = '    Output format: .H5 HDF-5 files\n';
    end
    SaveManager.logPrint(sp);
  
end

function created = locDirMaker(run, folderPath, infoStr)
%   This helper function is responsible for creating directories if testing for them finds that they
%   do not exist.
%
%>> savePath    full path to the save directory                                            str
%>> folder        folder to test for existance or create relative to savePath              str
%>> infoStr        an information string for reporting creation to the command line        str
%<< created        result specifying whether or not the director was created.              logical
        
    %--- Create folder is it doesn't exist. ---%
    created = exist(folderPath, 'dir'); %Test for folder exists
    if (created ~= 7)
        try mkdir(folderPath);
        catch MErr, created = MErr; return; end
        
        % no... let's not
	%--- Print creation results as feedback to the command line ---%
        if strcmp(run.paths.results, folderPath)
            displayPath = folderPath;
        else
            displayPath = strrep(folderPath, run.paths.results, '[Results Path]');
        end
        %fprintf('%s: %s\n', infoStr, displayPath);
        created = true;
    else
        created = false;
    end
    
end
