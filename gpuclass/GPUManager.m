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
        stackDeviceList, stackUseHalo, stackPartitionDir, stackCudaStreamsPtr, stackNprocs;
        numStack;
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
            g.numStack = 0;
        end

        function init(obj, devlist, halo, partitionDirection)
            if obj.isValidDeviceList(devlist)
                obj.deviceList = devlist;
            else
                
            end
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
        
        function pushParameters(self)
            % Puts current manager params on top of a stack,
            % permitting modification w/o risking loss of history.
            N = self.numStack + 1;
            self.numStack = N;
            self.stackDeviceList{N} = self.deviceList;
            self.stackUseHalo{N} = self.useHalo;
            self.stackPartitionDir{N} = self.partitionDir;
            self.stackCudaStreamsPtr{N} = self.cudaStreamsPtr;
            self.stackNprocs{N} = self.nprocs;
        end
        
        function popParameters(self)
            N = self.numStack;
            if N == 0; error('Cannot use popParameters before first pushParameters.'); end
            self.numStack = N-1;
            self.deviceList = self.stackDeviceList{N};
            self.useHalo = self.stackUseHalo{N};
            self.partitionDir = self.stackPartitionDir{N};
            self.cudaStreamsPtr = self.stackCudaStreamsPtr{N};
            self.nprocs = self.stackNprocs{N};
        end

    end%PUBLIC

%===================================================================================================        
    methods (Access = protected) %                                          P R O T E C T E D    [M]

        function tf = isValidDeviceList(self, devlist)
           mem = GPU_ctrl('memory');
           ndevs = size(mem,1);
           tf = all(devlist >= 0) && all(devlist < ndevs);
        end

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
