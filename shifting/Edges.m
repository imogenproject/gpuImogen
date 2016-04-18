classdef Edges < handle
% Class to handle edge related settings and functionality.
        
%===================================================================================================
        properties (Constant = true, Transient = true) %                                                        C O N S T A N T         [P]
                DIMENSION = {'x','y','z'};
                FIELDS = {'lower','upper'};
    end%CONSTANT
        
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                                                        P U B L I C  [P]
                ACTIVE;
                TOLERANCE;
                lower;
                upper;
    end %PUBLIC

%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                                   P R O T E C T E D [P]
                pIndex;
                pSlices;
    end %PROTECTED
        
    properties (SetAccess = protected, GetAccess = public)
        boundaryStatics;
    end      
        
%===================================================================================================
    methods %                                                                                                                                          G E T / S E T  [M]
        end%GET/SET
        
%===================================================================================================
    methods (Access = public) %                                                                                                                P U B L I C  [M]
                
%___________________________________________________________________________________________________ Edges
        function obj = Edges(bcModes, array, tolerance)
            if isempty(array)
                error(['Edges:EdgeError: Edge conditions set on empty array. ', ... 
                                        'Missing data may cause shifting errors.']);
            end
                
    %--- Initialize edge parameters ---%
            N               = size(array);
            dim             = ndims(array); % fine, waste bandwidth, just stfu
            obj.TOLERANCE   = tolerance;
                
            if (dim > 2)
                obj.pSlices = ones(3,2);
                obj.pIndex  = {1:N(1),1:N(2),1:N(3)};
            else
                obj.pSlices = ones(2,2);
                obj.pIndex  = {1:N(1),1:N(2)};
            end
            obj.pSlices(:,2) = size(array);
                
    %--- Store Edge ICs ---%
            obj.ACTIVE      = false(2, 3);

            S = StaticsInitializer(); % I only love you for your index calculator

            obj.boundaryStatics.index = [];
            obj.boundaryStatics.coeff = [];
            obj.boundaryStatics.value = [];
            obj.boundaryStatics(2) = obj.boundaryStatics(1);
            obj.boundaryStatics(3) = obj.boundaryStatics(1);

            for n=1:2
            for i=1:dim
                iIndex      = obj.pIndex;
                switch bcModes{n, i}
                    % Static BC: hold the three cells adjacent to that edge fixed to original value forever
                    case ENUM.BCMODE_STATIC
                        if (n == 1)
                            uslice = (1:3)+S.GIS.pLocalDomainOffset(i);
                        else
                            uslice = (-2:0) + size(array,i) + S.GIS.pLocalDomainOffset(i);
                        end

                        if (i == 1); indset = S.indexSetForVolume(uslice,[],[]); end
                        if (i == 2); indset = S.indexSetForVolume([],uslice,[]); end
                        if (i == 3); indset = S.indexSetForVolume([],[],uslice); end

%ne = size(indset, 1);
%fprintf('Rank %i: boundary in %i dir, side %i has %i elements\n', mpi_myrank(), i, n, ne);

                        obj.boundaryStatics(i).index = [obj.boundaryStatics(i).index; indset];
                        obj.boundaryStatics(i).coeff = [obj.boundaryStatics(i).coeff; ones([size(indset,1) 1]) ];
                        obj.boundaryStatics(i).value = [obj.boundaryStatics(i).value; array.array(indset(:,1)+1)];

                    case ENUM.BCMODE_TRANSPARENT
                        obj.ACTIVE(n,i) = true;

                        if (n == 1)
                            iIndex{i}   = 1;
                            field       = 'lower';
                        else
                            iIndex{i}   = size(array,i);
                            field       = 'upper';
                        end
                        
                        obj.(field).(Edges.DIMENSION{i}) = array(iIndex{:});
                        
                        if ~isa(array,'double') %r2009b: iscodistributed
                            obj.(field).(Edges.DIMENSION{i}) ...
                                                     = gather(obj.(field).(Edges.DIMENSION{i}));
                        end

                        obj.(field).(Edges.DIMENSION{i}) = squish(obj.(field).(Edges.DIMENSION{i}));

                    case ENUM.BCMODE_FADE
                        WIDTH=16;
                        if (n == 1)
                            uslice  = 1:WIDTH;
                            uprime = uslice;
                            yinterp = .2*(1 - (uslice - 3)/(WIDTH-2)); yinterp(1:3) = 1;
                        else
                            uslice  = ((1-WIDTH):0) + size(array,i);
                            uprime  = WIDTH:-1:1;
                            yinterp = .2*(1 - (uprime - 3)/(WIDTH-2)); yinterp((end-2):end) = 1;
                        end

                        if (i == 1); indset = S.indexSetForVolume(uslice,[],[]); end
                        if (i == 2); indset = S.indexSetForVolume([],uslice,[]); end
                        if (i == 3); indset = S.indexSetForVolume([],[],uslice); end

                        fadecoeff = interp1(uslice, yinterp, indset(:,i+1));

                        obj.boundaryStatics(i).index = [obj.boundaryStatics(i).index; indset];
                        obj.boundaryStatics(i).coeff = [obj.boundaryStatics(i).coeff; fadecoeff];
                        obj.boundaryStatics(i).value = [obj.boundaryStatics(i).value; array.array(indset(:,1)+1)];

                    case ENUM.BCMODE_WALL
                        obj.ACTIVE(n,i) = true;
                        
                        if (n == 1)
                            iIndex{i} = 1;
                            field     = 'lower';
                        else
                            iIndex{i} = N(i);
                            field     = 'upper';
                        end
                        
                        obj.(field).(Edges.DIMENSION{i}) = array(iIndex{:});
                end
            end
            end

            % We now 'compile' these as was done with the statics internal to the grid
            % FIXME: Fix the wall, fade and transparent BCs
            for i = 1:3
                if numel(obj.boundaryStatics(i).value) == 0; continue; end
                [obj.boundaryStatics(i).index obj.boundaryStatics(i).value obj.boundaryStatics(i).coeff] = staticsPrecompute(obj.boundaryStatics(i).index, obj.boundaryStatics(i).value, obj.boundaryStatics(i).coeff, S.GIS.pLocalRez);
%                [obj.boundaryStatics.index obj.boundaryStatics.value obj.boundaryStatics.coeff] = staticsPrecompute(obj.boundaryStatics.index, obj.boundaryStatics.value, obj.boundaryStatics.coeff, S.arrayDimensions);

            end
        end

%___________________________________________________________________________________________________ getEdge
        function result = getEdge(obj, upper, dim, array, bcType)
                    
        upper = upper + 1;
                    
        switch bcType
            case ENUM.BCMODE_TRANSPARENT
                iIndex                = obj.pIndex;
                iIndex{dim} = obj.pSlices(dim,upper);

                newEdge     = array(iIndex{:});
                if ~isa(array,'double') %r2009b: iscodistributed
                    newEdge = gather(newEdge);        
                end 
                
                newEdge     = squish(newEdge); 
                oldEdge     = obj.(Edges.FIELDS{upper}).(Edges.DIMENSION{dim});

                delta       = min(abs(newEdge - oldEdge),obj.TOLERANCE);
                signTest    = (newEdge - oldEdge) > 0;
                result      = oldEdge + (signTest - ~signTest) .* delta;

                obj.(Edges.FIELDS{upper}).(Edges.DIMENSION{dim}) = result; %Update edge
                                    
            case {ENUM.BCMODE_FADE, ENUM.BCMODE_WALL}
                result = obj.(Edges.FIELDS{upper}).(Edges.DIMENSION{dim});
            end
        end
                
    end%PUBLIC
        
%===================================================================================================        
    methods (Access = protected) %                                                                                        P R O T E C T E D    [M]
    end%PROTECTED
        
end%CLASS
