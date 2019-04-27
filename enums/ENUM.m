classdef ENUM
    
    properties (Constant = true)
        MASS      = 'mass';
        MOM       = 'mom';
        ENER      = 'ener';
        MAG       = 'mag';
        GRAV      = 'grav';
        
        SCALAR    = 0;
        VECTOR    = [1 2 3];
        
        POINT_FADE = 'point';
        
        GEOMETRY_SQUARE = 1;
        GEOMETRY_CYLINDRICAL = 2;
        
        %--- Pressure Types ---%
        PRESSURE_TOTAL_AND_SOUND = 'totsnd';
        PRESSURE_SOUND_SPEED     = 'sound';
        PRESSURE_GAS             = 'gas'
        PRESSURE_TOTAL           = 'total'
        PRESSURE_MAGNETIC        = 'magnetic'
        
        %--- Boundary Condition Modes  ---%
        % These are implemented and work as advertised
        BCMODE_CIRCULAR     = 'circ';       % Circular/periodic BC at edge
        BCMODE_CONSTANT     = 'const';      % Constant-value extrapolation; Appropriate for supersonic in/outflow
        BCMODE_LINEAR       = 'linear';     % Linear extrapolation; Dangerous, prone to backflow instability
        BCMODE_MIRROR       = 'mirror';     % Mirror BC (scalars, vector parallel = symmetry, vector perp = antisymmetry)
        BCMODE_STATIC       = 'bcstatic';   % Read values at t=0 and reset to this value at every step
        BCMODE_OUTFLOW      = 'outflow';
        BCMODE_FREEBALANCE  = 'freebalance';% Solves isothermal gravity balance normally and slip BC transversly

        % These not so much
        BCMODE_FADE         = 'fade';       % Fade arrays out to ICs at edges.
        BCMODE_FLIP         = 'flip';       % Flips vector boundaries along shifting direction.
        BCMODE_TRANSPARENT  = 'trans';      % Transparent boundary condition type.
        BCMODE_WALL         = 'wall';       % Immutable wall boundary set by initial conditions.
        BCMODE_ZERO         = 'zero';       % Zero fluxes but constant values.

        %--- Gravitational Solvers ---%
        GRAV_SOLVER_EMPTY       = 'empty';         % Empty solver.
        GRAV_SOLVER_BICONJ      = 'biconj';        % Linear solver using BICGSTAB.
        GRAV_SOLVER_GPU         = 'biconj_gpu';    % Linear solver using BICGSTAB running on GPU
        GRAV_SOLVER_MULTIGRID   = 'multigrid';     % Hierarchial discretization solver.
        
        GRAV_BCSOURCE_FULL      = 'full';          % Computes boundary conditions at every boundary cell
        GRAV_BCSOURCE_INTERP    = 'interpolated';  % Computes boundary conditions at every 4th boundary
                                                   % cell and interpolates

        % How many times denser than run.fluid.MINMASS before gravity is felt by the fluid
        GRAV_FEELGRAV_COEFF     = 4.0;

        %--- Radiation Model Types ---%
        RADIATION_NONE              = 'empty';
        RADIATION_OPTICALLY_THIN    = 'optically_thin';
        
        %--- Artificial Viscosity Types ---%
        ARTIFICIAL_VISCOSITY_NONE               = 'empty';
        ARTIFICIAL_VISCOSITY_NEUMANN_RICHTMYER  = 'neumann_richtmyer';
        ARTIFICIAL_VISCOSITY_CARAMANA_SHASHKOV_WHALEN = 'caramana_shashkov_whalen';

        CUATOMIC_SETMIN  = 1;
        CUATOMIC_SETMAX  = 2;
        CUATOMIC_FIXNAN  = 3;

        FORMAT_MAT  = 1;
        FORMAT_NC   = 2;
	FORMAT_HDF  = 3;
        
        CFD_HLL = 1;
        CFD_HLLC = 2;
        CFD_XINJIN = 3;

        MULTIFLUID_EMP = 0;
        MULTIFLUID_RK4 = 1;
        MULTIFLUID_ETDRK1 = 2;
        MULTIFLUID_ETDRK2 = 3;
        MULTIFLUID_LOGTRAP2 = 4;
	MULTIFLUID_LOGTRAP3 = 5;
    end

end
