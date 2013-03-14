classdef GravityManager < handle
% This is the management class for the potential solver. This is a singleton class to be accessed 
% using the getInstance() method and not instantiated directly. Currently the gravitational code is
% setup for a gravitational constant, G, of one.
%===================================================================================================
    properties (Constant = true, Transient = true) %                         C O N S T A N T     [P]
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public, Transient = true) %         P U B L I C  [P]
    ACTIVE;             % Specifies that gravity spolver state                          logical 
    info;               % Stores gravity solver report information                      str        

    laplacianMatrix;   % 4th order discrete Laplacian for gravity solver                sparse
    lowerConditioner;  % Incomplete LU factorizations to precondition the gravity       sparse
    upperConditioner;  % Solver for rapid solution                                      sparse

    constant;           % Gravitational scaling constant. (defaulted to 1)              double
    iterMax;            % Max number of iterations before gravity solver stops          double
    tolerance;          % Escape tolerance for the iterative solver                     double

    solve;              % Function handle to the potential solver                       handle
    solverInit;

    compactObjects;     % CompactObject{} 

    bconditionSource;   % Determines use of full or interpolated boundary conditions    string

    mirrorZ            % If true creates BCs with mass mirrored across lower XY plane  bool [false]

    array;

    TYPE;               % Gravity solver type enumeration                               str
    end%PUBLIC
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = private) %                         P R I V A T E [P]
        pMatrixActive;     % Specifies sparse matrix active/inactive                    logical
        parent;            % Manager parent                                             ImogenManger
    end %PROPERTIES 
    
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
    end%GET/SET    
    
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]        
        
%___________________________________________________________________________________________________ createSparseMatrix
% Builds a Laplacian and a preconditioner for linear solver methods. First creates the 6th order
% approximate Laplacian matrix and then performs a block incomplete LU factorization to create a
% preconditioner. The block size is chosen as Nx*Ny/2 as this has been found to be a good compromise
% between preconditioner size and facilitating rapid convergence.
function createSparseMatrix(obj, grid, dgrid)
            if (~obj.ACTIVE || ~obj.pMatrixActive); return; end

            fprintf('Generating subset of Laplacian matrix to build preconditioner...\n');
            
            blockSize = grid(1)*grid(2)/2;
            fprintf('Generating block ILU preconditioner, block size %i...\n', blockSize);
            [obj.lowerConditioner obj.upperConditioner] = ...
                poissonBlockILU(obj.laplacianMatrix, .05, blockSize, [prod(grid) prod(grid)]);

        end

%___________________________________________________________________________________________________ setSolver
% Attaches the correct solver function to the solve handle property as specified by the input type.
        function setSolver(obj, type)
            obj.TYPE = type;
            
            switch type
                %-----------------------------------------------------------------------------------
                case ENUM.GRAV_SOLVER_EMPTY
                    obj.solve           = @emptyPotentialSolver;
                    obj.solverInit      = @emptyPotentialSolverIni;
                    obj.pMatrixActive   = false;
                %-----------------------------------------------------------------------------------
                case ENUM.GRAV_SOLVER_BICONJ
                    obj.solve           = @bicgstabPotentialSolver;
                    obj.solverInit      = @bicgstabPotentialSolverIni;
                    obj.pMatrixActive   = true;
                case ENUM.GRAV_SOLVER_GPU
                    obj.solve           = @bicgstabPotentialSolver_GPU;
                    obj.solverInit      = @bicgstabPotentialSolverIni_GPU;
                    obj.pMatrixActive   = false;
               %-----------------------------------------------------------------------------------
                case ENUM.GRAV_SOLVER_MULTIGRID
                    obj.solve           = @multigridPotentialSolver;
                    obj.solverInit      = @multigridPotentialSolverIni;
                    obj.pMatrixActive   = false;
                %-----------------------------------------------------------------------------------
                otherwise
                    obj.type            = ENUM.GRAV_SOLVER_EMPTY;
                    obj.solve           = @emptyPotentialSolver;
                    obj.solverInit      = @emptyPotentialSolverIni;
                    obj.pMatrixActive   = false;
            end
                
            
        end
        
%___________________________________________________________________________________________________ solvePotential
% Actual method call for finding the gravitational potential for a given mass distribution. Solver
% has an initial abort statement that exits if the gravitational solver is not active for a run.
        function solvePotential(obj, mass)
            if ~obj.ACTIVE; obj.array = []; return; end
            
            if ~strcmp(obj.TYPE, ENUM.GRAV_SOLVER_EMPTY)
                obj.array = obj.solve(run, mass.array, mass.gridSize, 0);
            end

        end

        function initialize(obj, initialConditions, mass)

            obj.setSolver(initialConditions.type);
            obj.constant         = initialConditions.constant;
            obj.iterMax          = initialConditions.iterMax;
            obj.tolerance        = initialConditions.tolerance;
            obj.bconditionSource = initialConditions.bconditionSource;

            obj.mirrorZ          = initialConditions.mirrorZ;

            obj.solverInit(obj, mass);

            obj.compactObjects = [];

            for n = 1:size(initialConditions.compactObjectStates)
                % Compact object states:
                % [m R x y z vx vy vz lx ly lz]

                % Stellar state vector:
                % [x y z R px py pz lx ly lz M rho_v rho_g E_v]
                s  = initialConditions.compactObjectStates(n,:);
                M  = s(1);
                fluid = obj.parent.fluid;
                e0 = fluid.MINMASS*.02 * 3/10; % Awful hack for min pressure
                obj.addCompactObject([s(3) s(4) s(5) s(2) M*s(6) M*s(7) M*s(8) s(9) s(10) s(11) M fluid.MINMASS ENUM.GRAV_FEELGRAV_COEFF*fluid.MINMASS e0]);
            end

            if mpi_amirank0()
                run = ImogenManager.getInstance();
                run.save.logPrint(sprintf('Total of %i CompactObjects present on grid.\n', numel(obj.compactObjects)));
            end
        end

        function addCompactObject(obj, stateVector)
            obj.compactObjects{end+1} = CompactObject(stateVector);
        end

    end%PUBLIC
     
%===================================================================================================    
    methods (Access = private) %                                                P R I V A T E    [M]
        
%___________________________________________________________________________________________________ GravityManager
% Creates a new GravityManager instance and intializes it with default settings.
        function obj = GravityManager() 
            obj.setSolver( ENUM.GRAV_SOLVER_EMPTY );
            obj.ACTIVE      = false;
            obj.solverInit  = @grav_ini_nonGravitational;

            obj.tolerance   = 1e-10;
            obj.iterMax     = 100;
            obj.constant    = 1;

            obj.bconditionSource = ENUM.GRAV_BCSOURCE_FULL;
        end
        
    end%PROTECTED
        
%===================================================================================================    
    methods (Static = true) %                                                      S T A T I C    [M]
        
%___________________________________________________________________________________________________ getInstance
% Accesses the singleton instance of the GravityManager class, or creates one if none have
% been initialized yet.
        function singleObj = getInstance()
            persistent instance;
            if isempty(instance) || ~isvalid(instance) 
                instance = GravityManager();
            end
            singleObj = instance;
        end
        
    end%STATIC
    
end
