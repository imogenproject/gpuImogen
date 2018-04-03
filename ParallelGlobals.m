classdef ParallelGlobals < handle
   properties (GetAccess = public, SetAccess = private)
       context;
       topology;
   end
   
   methods
        function obj = ParallelGlobals(context, topology)
            % Acquires information about MPI, builds some basic contextual info,
            % and stores this in a global var. This is information that can
            % only really be got once.
            % FIXME: global variable alert! This is the one of these things I haven't gotten rid of yet.
            persistent instance;

            if ~(isempty(instance) || ~isvalid(instance))
                obj = instance; return;
            end

            if nargin < 2;
                warning('GlobalIndexSemantics received no topology: generating one');
                if nargin < 1;
                    warning('GlobalIndexSemantics received no context: generating one.');
                    obj.context = parallel_start();
                end
                obj.topology = parallel_topology(obj.context, 3);
            else
                obj.topology = topology;
                obj.context = context;
            end
            
            instance = obj;
        end

        function setNewTopology(self, topodim)
            self.topology = mpi_deleteDimcomm(self.topology); % wipe out MPI communicators
            self.topology = parallel_topology(self.context, topodim);
        end
   end
end
