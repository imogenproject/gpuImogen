classdef ODE_Matrix < handle
% Class annotation template for creating new classes.
%___________________________________________________________________________________________________ 

%===================================================================================================
        properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
        
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
    end %PUBLIC

%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
	matrixList;
	% MatrixList is a structure composed of the following fields:
	% matrixList.hash[]: Vector of integers which identify the methods
	% matrixList.coefficients[]: Vector of cells which contain the rational fractions which
	% give the exact coefficients of a method, consisting of [a b c d ... | LCD] which are all
	% integers from which method coefficients are [a/LCD b/LCD c/LCD ... ]
	dbFile;
	altered; % If anything was changed we need to save
    end %PROTECTED
    
%===================================================================================================
    methods (Access = public) %                                                                     G E T / S E T  [M]
	function me = ODE_Matrix(datafile, nouveau)
	    if nargin == 1
	        load(datafile); % Must contain a variable ML of the form given above for MatrixList
		me.matrixList = ML;
	        me.dbFile = datafile;
	    else
		me.matrixList.hash = 0;
		me.matrixList.coefficients = cell(1);
		me.dbFile = datafile;
	    end

	    me.altered = 0;
	end

	function delete(self)
	    if self.altered == 1
	        ML = self.matrixList;
	        save(self.dbFile, 'ML');    
	    end
	end

	function h = generateMethodHash(self, id)
	    reqs = (id ~= 0)*1.0;

	    x = sum(reqs, 2);
	    h = 0;
	    for i = 1:size(x)
		h = h + x(i)*32^i;
	    end
	    % This can handle up to 10 derivatives and 15 points...
	    % Generates a "hash" based on how many times a derivative must be matched
	    % E.G. if we match
	    % [1 1 1 1]
	    % [1 1 1  ]
	    % [1 1    ]
	    % Then x will be [4; 3; 2].
	    % Then we match f at 4 points, f' at 3 and f'' at 2 so we would have 4 + 3*32 + 2*1024 as our unique identifier
	    % This is predicated on never having "gaps" in the constraint table, i.e. always left-flushed. 
	end  

	function M = retreiveMethod(self, id)
	    if (numel(id) == 1) && (id >= 32) % Assue this is the hash itself
		hToGet = id;
	    else % Assume this is a constraint matrix
		hToGet = self.generateMethodHash(id);
	    end

	    x = find(self.matrixList.hash == hToGet);
	    if numel(x) > 1; warning('WARNING: MULTIPLE RESULTS FOR THIS METHOD!'); end
	    x=x(1);

	    pq = self.matrixList.coefficients{x};
	    M = pq(:,1:(end-1));
	    for i = 1:size(M,1); M(i,:) = M(i,:) / pq(i,end); end
	end

	function status = storeMethod(self, M)
	    numerator = M(:,1:(end-1));
	    hToStore = self.generateMethodHash(numerator);
	    self.matrixList.hash(end+1) = hToStore;
	    self.matrixList.coefficients{end+1} = M;

            status = 0;
	    self.altered = 1;
	end

	function status = removeMethod(self, M)
	    if numel(M) == 1
		hToKill = id;
	    else
		hToKill = self.generateMethodHash(M);
	    end

	    zap = ~(self.matrixList.hash == hToKill);
	    self.matrixList.hash = self.matrixList.hash(zap);
	    self.matrixList.coefficients = self.matrixList.coefficients(zap);

	    fprintf('Removed %i methods.\n', int32(nnz(~zap)));

	    self.altered = 1;
	end

	function about(self)
	    fprintf('I have %i methods in my database.\n', int32(numel(self.matrixList.hash)));
	end

    end%GET/SET
        
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
        end%PUBLIC
        
%===================================================================================================        
        methods (Access = protected) %                                      P R O T E C T E D    [M]
        end%PROTECTED
                
%===================================================================================================        
        methods (Static = true) %                                                 S T A T I C    [M]
        end%PROTECTED
        
end%CLASS
