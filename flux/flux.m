function flux(run, fluid, mag, order)
% This function manages the fluxing routines for the dimension-split convection code.
% By running through all possible splitting sequences it attempts to avoid the potential
% bias of any one Strang splitting.
%
%>< run         run manager object                                                      ImogenManager
%>< fluid       Array of fluid dynamic states                                           FluidManager(N)
%>< mag         magnetic field                                                          MagnetArray(3)
%>> order       direction of flux sweep (1 forward/-1 backward)                         int     +/-1

dims = run.geometry.globalDomainRez;

isOneDimensional = (numel(find(dims > 2)) == 1);
if isOneDimensional
    for gas = 1:size(fluid)
        relaxingFluid(run, fluid(gas).mass, fluid(gas).mom, fluid(gas).ener, mag, 1);
    end
    return;
end
%-----------------------------------------------------------------------------------------------
% Set flux direction and magnetic index components
%-------------------------------------------------

if dims(3) > 2 % three dimensional: We require 3 cells to be fluxable
    sweep = mod(run.time.iteration + 3*(order > 0),6)+1;
else           % two dimensional
    sweep = mod(run.time.iteration + (order<0), 2)+1;
end

% Any permutes that must be done before the sweep (not doing X first)
preperm = [0 2 0 2 3 3];

% The sequences of permutation & fluxing during the sweep
fluxcall = [1 2 1 2 3 3 ;3 1 2 3 1 2; 2 3 3 1 2 1];
permcall = [3 2 2 3 3 2; 2 3 3 5 2 6; 6 3 5 0 2 0];

%Any cases which require a second permute to restore indexes
% NOTE: added cudaArrayRotate options have eliminated this requirement.
%postperm = [3 0 2 0 0 0];

% p = [1 3 2; 2 1 3; 1 2 3; 2 3 1; 3 1 2; 3 2 1];
% magneticIndices = [3 2; 3 1; 2 1];
% magneticIndices = magneticIndices(directVec,:);
topo = run.geometry.topology;

% FI1ME: The subcalls should be rewritten to accept vectors of fluids...
for gas = 1:size(fluid)
    mass = fluid(gas).mass;
    mom  = fluid(gas).mom;
    ener = fluid(gas).ener;
    
    %===============================================================================================
    if (order > 0) %                             FORWARD FLUXING
    %===============================================================================================
        xchgIndices(run.pureHydro, mass, mom, ener, mag, preperm(sweep));
        for n = [1 2 3]
            % Skip identity operations
            if dims(fluxcall(n,sweep)) > 2
                relaxingFluid(run, mass, mom, ener, mag, fluxcall(n,sweep));
                xchgFluidHalos(mass, mom, ener, topo, fluxcall(n, sweep));
            end
            xchgIndices(run.pureHydro, mass, mom, ener, mag, permcall(n, sweep));
            % FIXME: magnetFlux has no idea these arrays may be permuted
            % FIXME: who cares, mhd in imogen is dead anyway
            %            if run.magnet.ACTIVE
            %                magnetFlux(run, mass, mom, mag, directVec(n), magneticIndices(n,[1 2]));
            %            end
        end
    %===============================================================================================
    else %                                       BACKWARD FLUXING
    %===============================================================================================
        xchgIndices(run.pureHydro, mass, mom, ener, mag, preperm(sweep));
        %        for n = [1 2 3];
        % FIXME: magnetFlux has no idea these arrays may be permuted
        % FIXME: who cares, mhd in imogen is dead anyway
        %            if run.magnet.ACTIVE
        %                magnetFlux(run, mass, mom, mag, directVec(n), magneticIndices(n,[2 1]));
        %            end
        for n = [1 2 3]
            % Skip identity operations
            if dims(fluxcall(n,sweep)) > 2;
                relaxingFluid(run, mass, mom, ener, mag, fluxcall(n,sweep));
                xchgFluidHalos(mass, mom, ener, topo, fluxcall(n, sweep));
            end
            xchgIndices(run.pureHydro, mass, mom, ener, mag, permcall(n, sweep));
            % FIXME: magnetFlux has no idea these arrays may be permuted
            % FIXME: who cares, mhd in imogen is dead anyway
            %            if run.magnet.ACTIVE
            %                magnetFlux(run, mass, mom, mag, directVec(n), magneticIndices(n,[1 2]));
            %            end
        end
    end
end

end

function xchgIndices(isFluidOnly, mass, mom, ener, mag, toex)
if toex == 0; return; end

s = { mass, ener, mom(1), mom(2), mom(3) };

for i = 1:5
    s{i}.arrayIndexExchange(toex, 1);
end

if isFluidOnly == 0
    s = {mag(1).cellMag, mag(2).cellMag, mag(3).cellMag};
    for i = 1:3
        s{i}.arrayIndexExchange(toex, 1);
    end
end

end

function xchgFluidHalos(mass, mom, ener, topology, dir)
s = { mass, ener, mom(1), mom(2), mom(3) };

for j = 1:5;
    cudaHaloExchange(s{j}, dir, topology, s{j}.bcHaloShare);
end

end

