classdef GPUManager < handle
% Class annotation template for creating new classes.
%___________________________________________________________________________________________________ 

%===================================================================================================
    properties (Constant = true, Transient = true) %                     C O N S T A N T         [P]
    end%CONSTANT

%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        deviceList; % Integers enumerating which GPUs this instance will use e.g. [0 2]
	useHalo;    % If > 0, boundaries between shared segments will have a useHalo-wide halo
        partitionDir;
        isInitd;
        cudaStreamsPtr;
	nprocs;
    end %PUBLIC

%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
        GIS;
    end %PROTECTED

%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
    end%GET/SET

%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
        function g = GPUManager()
	    g.deviceList = 0;
            g.useHalo = 0;
            g.partitionDir = 1;
            g.isInitd = 0;
            g.cudaStreamsPtr = int64(0); % initialize to NULL

            g.GIS = GlobalIndexSemantics();
            
            g.init([0], 3, 1);
        end

        function init(obj, devlist, halo, partitionDirection)
	    obj.deviceList = devlist;
% This was added when I thought it was needed, now it isn't.
%            obj.cudaStreamsPtr = GPU_ctrl('createStreams',devlist);
	    obj.useHalo = halo;
            obj.partitionDir = partitionDirection;
            obj.isInitd = 1;
	    obj.nprocs = obj.GIS.topology.nproc;
        end

        function describe(obj)
            fprintf('Current scheme uses devices ');
            for N = 1:numel(obj.deviceList)
                fprintf('%i ', int32(obj.deviceList(N)));
            end
            fprintf('.\nHalo size is %i, partitioning occurs in the %i direction.\n', int32(obj.useHalo), int32(obj.partitionDir));
        end

    end%PUBLIC

%===================================================================================================        
    methods (Access = protected) %                                          P R O T E C T E D    [M]
    end%PROTECTED

%===================================================================================================    
    methods (Static = true) %                                                      S T A T I C   [M]

%_______________________________________________________________________________________ getInstance
% Accesses the singleton instance of the ImogenManager class, or creates one if none have
% been initialized yet.
        function singleObj = getInstance()
            persistent instance;
            if isempty(instance) || ~isvalid(instance)
                instance = GPUManager();
            end
            singleObj = instance;
        end

    end%STATIC

end
