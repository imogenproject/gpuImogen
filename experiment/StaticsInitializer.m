classdef StaticsInitializer < handle
%___________________________________________________________________________________________________ 
%===================================================================================================
    properties (Constant = true, Transient = true) %                     C O N S T A N T         [P]
        CELLVAR = 0;
        FLUXL   = 1;
        FLUXR   = 2;
        FLUXALL = 3;
    end%CONSTANT
        
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        indexSet; % Arrays in indices to set statically            Cell[N]
        valueSet; % Arrays of values to set statically             Cell[N]
        coeffSet; % Arrays of coefficients to set [must match dimensions of corresponding value set]

        arrayStatics; % Cell array with one cell per simulation var Cell[8];
        % WARNING: THIS MUST BE THE SAME SIZE AS THE NUMBER OF SIMULATION VARIABLES

        geometry; % handle to the global semantics class

    end %PUBLIC

%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
        readyForReadout; % Set if prepareStaticsForSimulation() has been called and
                         % no new associations have been created since

    end %PROTECTED
        
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]

    end%GET/SET
        
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]

        function obj = StaticsInitializer(geometry)
            %>> geometry: GeometryManager class
            
            obj.geometry = geometry;
            obj.arrayStatics = cell(8,1); % Create one arrayStatics for every variable
            for x = 1:8
                obj.arrayStatics{x} = struct('arrayField',[], 'indexId',[], 'valueId',[], 'coeffId',[]);
                % Create structs to associate pairs of index sets and values with the primary & flux
                % arrays of each simulation variable
            end

        end

        function [indices, values, coeffs] = staticsForVariable(obj, varId, component, fieldId)
            %>> varId:     ...
            %>> component: ...
            %>> fieldId:   ...
            indices = [];
            values = [];
            coeffs = [];

            if ~obj.readyForReadout; obj.prepareStaticsForSimulation(); end

            partmap = obj.mapVaridToIdx(varId, component);
            if partmap == 0; return; end

            AS = obj.arrayStatics{partmap};

            for x = 1:numel(AS.arrayField) % For each static defined for this variable
                if AS.arrayField(x) == fieldId % If it applies to the requested field

                    newIdx   = obj.indexSet{AS.indexId(x)};
                    newVal   = obj.valueSet{AS.valueId(x)};

                    if AS.coeffId(x) == 0
                        newCoeff = 1;
                    else
                        newCoeff = obj.coeffSet{AS.coeffId(x)};
                    end

                    % Expand array-scalar pairs to array-array pairs; This can be done
                    if numel(newVal) == 1; newVal   = newVal   * ones(size(newIdx,1),1); end
                    if numel(newCoeff)==1; newCoeff = newCoeff * ones(size(newIdx,1),1); end

                    % Fail if nonequally sized arrays are paired; This cannot be done
                    if size(newVal,1) ~= size(newIdx,1)
                        error('Unrecoverable error preparing statics; numel(index set %i) = %i but numel(value set %i) = %i.\n', x, size(newIdx,1), x, numel(newVal));
                    end

                    indices = [indices; newIdx]; % cat indices
                    values  = [values ; newVal]; % cat values
                    coeffs  = [coeffs ; newCoeff]; % cat fade coefficients
                end
            end

        end

        % This function prepares statics for injection into array statics by reshaping for concatenation
        function prepareStaticsForSimulation(obj)
            % Reshape them to be Nx1 arrays so we can cat using [u; v]
            for x = 1:numel(obj.indexSet)
                % Reshape them to be Nx1
%                obj.indexSet{x} = reshape(obj.indexSet{x}, [numel(obj.indexSet{x}) 1]);
%                obj.valueSet{x} = reshape(obj.valueSet{x}, [numel(obj.valueSet{x}) 1]);
            end

            obj.readyForReadout = 1;

        end

        % Adds a pair of statics
        function addStatics(obj, indices, values, coeffs)
            obj.indexSet{end+1} = indices;
            obj.valueSet{end+1} = values;
            if nargin == 3
                obj.coeffSet{end+1} = 1;
            else
                obj.coeffSet{end+1} = coeffs;
            end
        end

        % Maps a set of indices, values and fade rate coefficients to a variable
        function associateStatics(obj, varID, component, fieldID, indexNum, valueNum, coeffNum)
            vmap = obj.mapVaridToIdx(varID, component);

            obj.arrayStatics{vmap}.arrayField(end+1) = fieldID;
            obj.arrayStatics{vmap}.indexId(end+1)    = indexNum;
            obj.arrayStatics{vmap}.valueId(end+1)    = valueNum;
            if nargin == 6
                obj.arrayStatics{vmap}.coeffId(end+1) = 0;
            else
                obj.arrayStatics{vmap}.coeffId(end+1)    = coeffNum;
            end

            obj.readyForReadout = 0;
        end

        %%% === Utility functions === %%%

        function setFluid_allConstantBC(obj, mass, ener, mom, facenumber)
            obj.setConstantBC(ENUM.MASS, ENUM.SCALAR,   obj.CELLVAR, mass, facenumber);
            obj.setConstantBC(ENUM.ENER, ENUM.SCALAR,   obj.CELLVAR, ener, facenumber);
            obj.setConstantBC(ENUM.MOM, ENUM.VECTOR(1), obj.CELLVAR, squish(mom(1,:,:,:)), facenumber);
            obj.setConstantBC(ENUM.MOM, ENUM.VECTOR(2), obj.CELLVAR, squish(mom(2,:,:,:)), facenumber);
            if size(mom,4) > 1
                obj.setConstantBC(ENUM.MOM, ENUM.VECTOR(3), ENUM.CELLVAR, squish(mom(3,:,:,:)), facenumber);
            end

        end

        function setMag_allConstantBC(obj, mag, facenumber)
            obj.setConstantBC(ENUM.MAG, ENUM.VECTOR(1), obj.CELLVAR, squish(mag(1,:,:,:)), facenumber);
            obj.setConstantBC(ENUM.MAG, ENUM.VECTOR(2), obj.CELLVAR, squish(mag(2,:,:,:)), facenumber);
            if size(mag,4) > 1
                obj.setConstantBC(ENUM.MAG, ENUM.VECTOR(3), ENUM.CELLVAR, squish(mag(3,:,:,:)), facenumber);
            end
        end

        function setFluid_allFadeBC(obj, mass, ener, mom, facenumber, bcinf)
            obj.setFadeBC(ENUM.MASS, ENUM.SCALAR,   obj.CELLVAR, mass, facenumber, bcinf);
            obj.setFadeBC(ENUM.ENER, ENUM.SCALAR,   obj.CELLVAR, ener, facenumber, bcinf);
            obj.setFadeBC(ENUM.MOM, ENUM.VECTOR(1), obj.CELLVAR, squish(mom(1,:,:,:)), facenumber, bcinf);
            obj.setFadeBC(ENUM.MOM, ENUM.VECTOR(2), obj.CELLVAR, squish(mom(2,:,:,:)), facenumber, bcinf);
            if size(mom,4) > 1
                obj.setFadeBC(ENUM.MOM, ENUM.VECTOR(3), ENUM.CELLVAR, squish(mom(3,:,:,:)), facenumber, bcinf);
            end
        end

        function setMag_allFadeBC(obj, mag, facenumber, bcinf)
            obj.setFadeBC(ENUM.MAG, ENUM.VECTOR(1), obj.CELLVAR, squish(mag(1,:,:,:)), facenumber, bcinf);
            obj.setFadeBC(ENUM.MAG, ENUM.VECTOR(2), obj.CELLVAR, squish(mag(2,:,:,:)), facenumber, bcinf);
            if size(mag,4) > 1
                obj.setFadeBC(ENUM.MAG, ENUM.VECTOR(3), ENUM.CELLVAR, squish(mag(3,:,:,:)), facenumber, bcinf);
            end
        end

        function setConstantBC(obj, varID, component, fieldID, array, facenumber)
            %vmap = obj.mapVaridToIdx(varID, component);

            xset=[]; yset=[]; zset=[];

            switch facenumber
                case 1; xset=1:2;
                        yset=1:size(array,2); zset=1:size(array,3); % minus X
                case 2; xset=(size(array,1)-1):size(array,1);
                        yset=1:size(array,2); zset=1:size(array,3);% plux  X

                case 3; xset=1:size(array,1); % minus Y
                        yset=1:2; zset=1:size(array,3);
                case 4; xset=1:size(array,1); % plus  Y
                        yset=(size(array,2)-1):size(array,2); zset=1:size(array,3);

                case 5; xset=1:size(array,1); yset=1:size(array,2); % minus Z
                        zset=1:2;
                case 6; xset=1:size(array,1); yset=1:size(array,2); % plus  Z
                        zset=(size(array,3)-1):size(array,3);
            end

            inds = obj.indexSetForVolume(xset, yset, zset);

            obj.addStatics(inds, array(inds(:,1)));

            obj.associateStatics(varID, component, fieldID, numel(obj.indexSet), numel(obj.valueSet), numel(obj.coeffSet));
        end

    %%%%% ============== Assistant for boundary conditions setup =============== %%%%%
        function indices = indexSetForCube(obj, xslice, yslice, zslice)
            % Computes linear (in sane, zero indexed form that will be used by GPU routines)
            % indices of the cube
            % >> xslice: vector of global x coordinates, in Matlab count-from-1
            % >> yslice: vector of global y coordinates
            % >> zslice: vector of global z coordinates
            % << indices: [linear localx localy localz] columns of coordinates; linear is index from 0, others in Matlab count-from-1
            
            S = {xslice, yslice, zslice};

            for dim = 1:3
                if isempty(S{dim})
                    s0 = (1:obj.geometry.localDomainRez(dim));
                else
                    s0 = S{dim} - obj.geometry.pLocalDomainOffset(dim);
                end
                s0 = s0( (s0 > 0) & (s0 <= obj.geometry.localDomainRez(dim)) );
                S{dim} = s0;        
            end
                
            % Build the grid and compute linear offset indices for it IN GPU ADDRESSES (0-indexed)
            [u, v, w] = ndgrid(S{:});

            indices = (u-1) + obj.geometry.localDomainRez(1)*((v-1) + obj.geometry.localDomainRez(2)*(w-1));
            indices = [indices(:) u(:) v(:) w(:)];
        end

        function indices = indexSetForLogical(obj, xbound, ybound, zbound, truth)
            % >> xbound: two element [min max] specifying the range of 
        end
        
        function indices = indexSetForFunction(obj, func, xbound, ybound, zbound)
            xs = xbound(1):xbound(2);
            ys = ybound(1):ybound(2);
            zs = zbound(1):zbound(2);
            
            [u, v, w] = ndgrid(xs, ys, zs);
            
            truth = arrayfun(func, u, v, w);
            
            u = u(truth==1); v = v(truth==1); w = w(truth==1);
            
            [inds] = obj.geometry.toLocalIndices([u(:) v(:) w(:)]);
            a=inds(:,1); b = inds(:,2); c = inds(:,3);
            
            indices = (a-1)+obj.geometry.localDomainRez(1)*( (b-1) + obj.geometry.localDomainRez(2)*(c-1));
            indices = [indices(:) u(:) v(:) w(:)];
        end


    end%PUBLIC
        
%===================================================================================================        
    methods (Access = protected) %                                          P R O T E C T E D    [M]

    end%PROTECTED
                
%===================================================================================================        
    methods (Static = true) %                                                     S T A T I C    [M]

        function result = mapVaridToIdx(varId, component)
            if strcmp(varId,ENUM.MASS); result = 1; return; end
            if strcmp(varId,ENUM.ENER); result = 2; return; end
            if strcmp(varId,ENUM.MOM); result = 2+component; return; end % 3-4-5
            if strcmp(varId,ENUM.MAG); result = 5+component; return; end % 6-7-8

            result = 0;
            return;
        end

    end%PROTECTED
        
end%CLASS
