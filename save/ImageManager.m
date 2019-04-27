classdef ImageManager < handle
    % The manager class responsible for handling image related actions (primarily saving).
    
    %===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T     [P]
        IMGTYPES = {'mass','ener','momX','momY','momZ','magX', ... % The possible image types
            'magY','magZ','grav','spen','pGas', 'pTot',...
            'mach','speed','velX','velY','velZ'};
    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = public, GetAccess = public, Transient = true) %            P U B L I C  [P]
        ACTIVE;        % Specifies if image saving is active for a run.               boolean
        INTERVAL;    % Interval between image saves.                                  int
        COLORMAP;    % Colormap to use for the images.                                double
        frame;        % Frame index number for the next image save operation.         int
        
        
        %--- Specify whether or not the saving of an image type is allowed ---%       boolean
        mass;   ener;   spen;
        momX;   momY;   momZ;
        magX;   magY;   magZ;
        pGas;   pTot;    grav;
        velX;   velY;   velZ;
        mach;   speed;
        
        logarithmic;  % Contains fields to save as logarithmic images.                struct

        parallelUniformColors; % logical: Do or do not determine the color scaling based on global rather
                               % than local min/max values
    end%PUBLIC
    
    %===================================================================================================
    properties (SetAccess = public, GetAccess = protected) %                   P R O T E C T E D [P]
        pColordepth;    % # of color values to use in creating the colormaps          int
        parent;            % parent manager                                           ImogenArray
        pActive;        % active image saving slices
    end %PROTECTED
    
    %===================================================================================================
    methods %                                                                      G E T / S E T  [M]
    end%GET/SET
    
    %===================================================================================================
    methods (Access = public) %                                                        P U B L I C  [M]
        %__________________________________________________________________________________ ImageManager
        % Creates a new ImageManager instance.
        function obj = ImageManager()
            obj.frame = 0;
            for i=1:length(obj.IMGTYPES),     obj.(obj.IMGTYPES{i}) = false;    end
        end
        
        function parseIni(self, ini)
           
            fields = self.IMGTYPES;
            for i=1:length(fields)
                if isfield(ini.image,fields{i})
                    self.(fields{i}) = ini.image.(fields{i});
                end
                if isfield(ini.image,'logarithmic') && isfield(ini.image.logarithmic,fields{i})
                    self.logarithmic.(fields{i}) = ini.image.logarithmic.(fields{i});
                end
            end
            self.activate();
            
            if self.ACTIVE
                
                if isfield(ini.image,'interval')
                    self.INTERVAL = max(1,ini.image.interval);
                else
                    self.INTERVAL = 1;
                    self.parent.appendWarning('Image saving interval set to every step.');
                end
                
                if isfield(ini.image,'colordepth');    colordepth = ini.image.colordepth;
                else;                                  colordepth = 256;
                end
                
                if isfield(ini.image,'colormap'); self.createColormap(ini.image.colormap, colordepth);
                else;                                 self.createColormap('jet',colordepth);
                end
                
                imageSaveState = 'Active'; % FIXME: Wh... why is this a string?
            else; imageSaveState = 'Inactive';
            end
            self.parent.appendInfo('Image saving is', imageSaveState);
            
            if isfield(ini.image,'parallelUniformColors')
                self.parallelUniformColors = ini.image.parallelUniformColors; end
        end
        
        %____________________________________________________________________________________ initialize
        % Preliminary actions setting up image saves for the run. Determines which image slices to save.
        function initialize(obj)
            obj.pActive = obj.parent.save.ACTIVE(4:6);
            if ~any(obj.pActive)
                [minval, mindex] = min(obj.parent.geometry.globalDomainRez); %#ok<ASGLU>
                switch mindex
                    case 1;        obj.pActive(3) = true;
                    case 2;        obj.pActive(2) = true;
                    case 3;        obj.pActive(1) = true;
                end
            end
        end
        
        %______________________________________________________________________________________ activate
        % Activates the ImageManager by determining if any images have been enabled for saving.
        function activate(obj)
            obj.ACTIVE = false;
            for i=1:length(obj.IMGTYPES),    obj.ACTIVE = ( obj.ACTIVE || obj.(obj.IMGTYPES{i}) ); end
        end
        
        %___________________________________________________________________________________ getColormap
        function createColormap(obj, type, colordepth)
            obj.pColordepth = colordepth;
            switch (type)
                case 'jet';     obj.COLORMAP = jet(colordepth);
                case 'hot';     obj.COLORMAP = hot(colordepth);
                case 'bone';    obj.COLORMAP = bone(colordepth);
                case 'copper';  obj.COLORMAP = copper(colordepth);
                otherwise;      obj.COLORMAP = jet(colordepth);
            end
        end
        
        %______________________________________________________________________________ imageSaveHandler
        % Handles saving of images to files.
        function imageSaveHandler(obj, mass, mom, ener, mag, grav)
            if ~( obj.ACTIVE && ~mod(obj.parent.time.iteration, obj.INTERVAL) ); return; end
            for i=4:6 % For each possible 2D slice
                if ~obj.pActive(i-3), continue; end
                
                for j=1:length(ImageManager.IMGTYPES)
                    if ~obj.(ImageManager.IMGTYPES{j}); continue; end
                    
                    switch ImageManager.IMGTYPES{j}
                        case 'mass'
                            array = obj.parent.save.getSaveSlice(mass.array,i);
                        case 'momX'
                            array = obj.parent.save.getSaveSlice(mom(1).array,i);
                        case 'momY'
                            array = obj.parent.save.getSaveSlice(mom(2).array,i);
                        case 'momZ'
                            array = obj.parent.save.getSaveSlice(mom(3).array,i);
                        case 'ener'
                            array = obj.parent.save.getSaveSlice(ener.array,i);
                        case 'magX'
                            array = obj.parent.save.getSaveSlice(mag(1).array,i);
                        case 'magY'
                            array = obj.parent.save.getSaveSlice(mag(2).array,i);
                        case 'magZ'
                            array = obj.parent.save.getSaveSlice(mag(3).array,i);
                        case 'grav'
                            array = obj.parent.save.getSaveSlice(grav.array,i);
                            
                        case 'spen'
                            array = ener.array ./ mass.array;
                            array = obj.parent.save.getSaveSlice(array,i);
                            
                        case 'pTot'
                            a = pressure('total', obj.parent, mass, mom, ener, mag);
                            array = GPU_cudamemcpy(a); GPU_free(a);
                            array = obj.parent.save.getSaveSlice(array,i);
                            
                        case 'pGas'
                            a = pressure('gas', obj.parent, mass, mom, ener, mag);
                            array = GPU_cudamemcpy(a); GPU_free(a);
                            array = obj.parent.save.getSaveSlice(array,i);
                        case 'mach'
                            array = getMach(mass, mom, ener, mag, obj.parent.GAMMA);
                            array = obj.parent.save.getSaveSlice(array,i);
                            
                        case 'speed'
                            array = sqrt(getVelocitySquared(mass,mom));
                            array = obj.parent.save.getSaveSlice(array,i);
                            
                        case 'velX'
                            array = obj.parent.save.getSaveSlice(mom(1).array./mass.array,i);
                            
                        case 'velY'
                            array = obj.parent.save.getSaveSlice(mom(2).array./mass.array,i);
                            
                        case 'velZ'
                            array = obj.parent.save.getSaveSlice(mom(3).array./mass.array,i);
                            
                    end
                    
                    name = ImageManager.IMGTYPES{j};
                    
                    if isfield(obj.logarithmic,ImageManager.IMGTYPES{j})
                        logArray = ImageManager.findLog(array); % Convert to log w/value clamping
                        logArray = obj.parent.save.getSaveSlice(logArray,i);
                        logName = sprintf('log_%s',ImageManager.IMGTYPES{j});                                  
                        obj.saveImage(logArray, logName, obj.parent.save.SLICELABELS{i});                
                    end
                    
                    obj.saveImage(array, name, obj.parent.save.SLICELABELS{i});                
                end
            end
            obj.frame = obj.frame + 1;
        end

        
        
    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                            P R O T E C T E D    [M]
        
        %_____________________________________________________________________________________ saveImage
        %   This helper routine is responsible for saving image files. The default format here is an 8bit
        %   RGB png file.
        % array         array slice to write to an image file                           double  [nx ny]
        % name            name of the array for folder and file                           str     *
        % sliceType     slice type identifier to include in file name (eg. XY or YZ)    str     *
        function saveImage(obj, array, name, sliceType)
            parallels = ParallelGlobals(); % FIXME ugh this again
            minVal = min(min(array));
            maxVal = max(max(array));

            if obj.parallelUniformColors == true
                 glob = mpi_allgather([minVal maxVal]);
                 minVal = min(glob(1:2:end));
                 maxVal = max(glob(2:2:end));
            end
            
            rescaledArray = obj.pColordepth * (array' - minVal) / (maxVal - minVal);

            iterStr = obj.parent.paths.iterationToString(obj.frame);
            fileName = strcat(name,'_',sliceType,'_',sprintf('rank_%i_',parallels.context.rank),iterStr,'.png');
            filePathName = strcat(obj.parent.paths.image,filesep,name,filesep,fileName);
            imwrite(rescaledArray, obj.COLORMAP, filePathName, 'png', 'CreationTime', ...
                datestr(clock,'HH:MM mm-dd-yyyy'));
        end
        
        
    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %                                                      S T A T I C    [M]
        
        %_______________________________________________________________________________________________
        % Find natural log of absolute value, replacing infinities with next greater finite minimum.
        function result = findLog(array)           
            newMin = min(log(abs(nonzeros(array))));
            result = log(abs(array));    
            infinities = isinf(result);
            [I,J] = find(infinities == true);
            for k = 1:length(I)
                result(I(k),J(k)) = newMin;
            end      
        end        
        
    end%STATIC
end
