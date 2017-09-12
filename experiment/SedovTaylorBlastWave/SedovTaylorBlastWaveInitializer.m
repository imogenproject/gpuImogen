classdef SedovTaylorBlastWaveInitializer < Initializer
% Sets up a Sedov-Taylor explosion, consisting of a point deposition of energy in a near-zero-
% pressure background which drives a maximally symmetric explosion outward.
% c.f. Sedov (1959) for original solution or Kamm & Timmes (2007) for modern description of
% exact solution 1- to 3-D & multiple background densities.
        
%===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        autoEndtime;
        backgroundDensity;
        sedovAlphaValue;
        sedovExplosionEnergy;
    end %PUBLIC

%===================================================================================================
    properties (Dependent = true) %                                            D E P E N D E N T [P]
    end %DEPENDENT
    
%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
        pDepositRadius; % Radius (in cell) inside which energy will be deposited
        pBlastEnergy;

        pSedovAlpha;
    end %PROTECTED
    
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        
%___________________________________________________________________ SedovTaylorBlastWaveInitializer
        function obj = SedovTaylorBlastWaveInitializer(input)
            obj                  = obj@Initializer();
            obj.gamma            = 1.4;
            obj.runCode          = 'ST_BLAST';
            obj.info             = 'Sedov-Taylor blast wave trial.';
            obj.mode.fluid       = true;
            obj.mode.magnet      = false;
            obj.mode.gravity     = false;
            obj.cfl              = 0.4;
            obj.iterMax          = 10000;
            obj.bcMode.x         = ENUM.BCMODE_CONSTANT;
            obj.bcMode.y         = ENUM.BCMODE_CONSTANT;
            obj.bcMode.z         = ENUM.BCMODE_CONSTANT;
            obj.activeSlices.xy  = true;
            obj.activeSlices.xyz = true;
            obj.ppSave.dim2      = 5;
            obj.ppSave.dim3      = 20;
            obj.pureHydro = true;

            obj.depositRadiusCells(3.5);
            obj.setBlastEnergy(1);

            obj.backgroundDensity = 1;

            obj.autoEndtime = 1;

            obj.operateOnInput(input, [64 64 64]);
        end
               
        
    end%GET/SET
    
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]

        function depositRadiusCells(self, N); self.pDepositRadius = N; end
        function depositRadiusFraction(self, f); self.pDepositRadius = f*max(self.grid)/2; end
        function setBlastEnergy(self, Eblast)
            self.pBlastEnergy = Eblast;
         end


    end%PUBLIC
    
%===================================================================================================    
    methods (Access = protected) %                                          P R O T E C T E D    [M]

%________________________________________________________________________ calculateInitialConditions
        function [fluids, mag, statics, potentialField, selfGravity] = calculateInitialConditions(obj)

            geom = obj.geomgr;
            geom.makeBoxSize([1 1 1]);
	    geom.makeBoxOriginCoord(floor(geom.globalDomainRez/2 + 0.5));
            
            %--- Initialization ---%
            statics         = [];
            potentialField  = [];
            selfGravity     = [];

            if geom.globalDomainRez(3) == 1
                obj.bcMode.z = 'circ';
            end

            [mass, mom, mag, ener] = geom.basicFluidXYZ();

            mass            = mass * obj.backgroundDensity;

            obj.pSedovAlpha = SedovSolver.findAlpha(obj.pBlastEnergy, obj.backgroundDensity, obj.gamma, 2+1*(obj.geomgr.globalDomainRez(3)>1));

            if obj.autoEndtime
                if mpi_amirank0(); disp('Automatic end time selected: will run to blast radius = 0.45 (grid cube is normalized to size of 1)'); end
                obj.timeMax = SedovSolver.timeUntilSize(obj.pBlastEnergy, .45, obj.backgroundDensity, obj.gamma, 2+1*(obj.geomgr.globalDomainRez(3)>1), obj.pSedovAlpha);
            end

            %--- Calculate Radial Distance ---%
            [X, Y, Z] = geom.ndgridSetIJK('pos');
            distance  = sqrt(X.*X/(geom.d3h(1)^2) + Y.*Y/(geom.d3h(2)^2) + Z.*Z/(geom.d3h(3)^2));

            %--- Find the correct energy density Ec = Eblast / (# deposit cells) ---%
            % FIXME: Note that this is still 'wrong' in that it fails to integral average
            % However the error lies in high-harmonics of the distribution,
            % not the actual amount of energy - that is exactly correct and is most important for
            % Seeing to it that the 
            % WARNING: This works because we know from above that the coordinate zero is integer...
            gv = floor(-obj.pDepositRadius-1):ceil(obj.pDepositRadius+1);
            if obj.geomgr.globalDomainRez(3) == 1 % 2D
                [xctr, yctr] = ndgrid(gv,gv);
                activeCells = (xctr.^2+yctr.^2 < obj.pDepositRadius^2);
            else % 3D
                [xctr, yctr, zctr] = ndgrid(gv,gv,gv);
                activeCells = (xctr.^2+yctr.^2+zctr.^2 < obj.pDepositRadius^2);
            end
 
            nDepositCells = numel(find(activeCells));

            obj.sedovAlphaValue = obj.pSedovAlpha; % Public parameter that will be saved
            obj.sedovExplosionEnergy = obj.pBlastEnergy;
            
            %--- Determine Energy Distribution ---%
            ener            = 1e-8*mass/(obj.gamma-1); % Default to approximate zero pressure
            ener(distance < obj.pDepositRadius) = obj.pBlastEnergy / (nDepositCells*prod(obj.geomgr.d3h));

            fluids = obj.rhoMomEtotToFluid(mass, mom, ener);
        end
    
    end%PROTECTED
        
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
