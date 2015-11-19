function flux(run, mass, mom, ener, mag, order)
% This function manages the fluxing routines for the split code by managing the appropriate fluxing 
% order to try and average out any biasing caused by the Strang splitting.
%
%>< run         run manager object                                                      ImogenManager
%>< mass        mass density                                                            FluidArray
%>< mom         momentum density                                                        FluidArray(3)
%>< ener        energy density                                                          FluidArray
%>< mag         magnetic field                                                          MagnetArray(3)
%>> order       direction of flux sweep (1 forward/-1 backward)                         int     +/-1

    %-----------------------------------------------------------------------------------------------
    % Set flux direction and magnetic index components
    %-------------------------------------------------    

    if mass.gridSize(3) > 1
        sweep = mod(run.time.iteration-1 + 3*(order > 0),6)+1;
    else
        sweep = mod(run.time.iteration-1 + (order<0), 2)+1;
    end

    % Any permutes that must be done before the sweep (not doing X first)
    preperm = [0 2 0 2 3 3];

    % The sequences of permutation & fluxing during the sweep
    fluxcall = [1 2 1 2 3 3 ;3 1 2 3 1 2; 2 3 3 1 2 1];
    permcall = [3 2 2 3 3 2; 2 3 3 5 2 6; 6 3 5 0 2 0];

    % Any cases which require a second permute to restore indexes
    postperm = [3 0 2 0 0 0];

   % p = [1 3 2; 2 1 3; 1 2 3; 2 3 1; 3 1 2; 3 2 1];
   % magneticIndices = [3 2; 3 1; 2 1];
   % magneticIndices = magneticIndices(directVec,:);

    %===============================================================================================
    if (order > 0) %                             FORWARD FLUXING
    %===============================================================================================
        xchgIndices(run.pureHydro, mass, mom, ener, mag, preperm(sweep));
        for n = [1 2 3]
            % Skip identity operations
            if mass.gridSize(fluxcall(n,sweep)) > 3; 
                relaxingFluid(run, mass, mom, ener, mag, fluxcall(n,sweep));
                xchgFluidHalos(mass, mom, ener, fluxcall(n, sweep));
            end
            xchgIndices(run.pureHydro, mass, mom, ener, mag, permcall(n, sweep));
% FIXME: magnetFlux has no idea these arrays may be permuted
% FIXME: who cares, mhd in imogen is dead anyway
%            if run.magnet.ACTIVE
%                magnetFlux(run, mass, mom, mag, directVec(n), magneticIndices(n,[1 2]));
%            end
        end

%	xchgIndices(run.pureHydro, mass, mom, ener, mag, postperm(sweep));

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
            if mass.gridSize(fluxcall(n,sweep)) > 3;
                relaxingFluid(run, mass, mom, ener, mag, fluxcall(n,sweep));
                xchgFluidHalos(mass, mom, ener, fluxcall(n, sweep));
            end
            xchgIndices(run.pureHydro, mass, mom, ener, mag, permcall(n, sweep));
% FIXME: magnetFlux has no idea these arrays may be permuted
% FIXME: who cares, mhd in imogen is dead anyway
%            if run.magnet.ACTIVE
%                magnetFlux(run, mass, mom, mag, directVec(n), magneticIndices(n,[1 2]));
%            end
        end

 %       xchgIndices(run.pureHydro, mass, mom, ener, mag, postperm(sweep));
    end

end

function xchgIndices(isFluidOnly, mass, mom, ener, mag, toex)
if toex == 0; return; end

s = { mass, ener, mom(1), mom(2), mom(3) };

for i = 1:5
    s{i}.arrayIndexExchange(toex, 1);
    s{i}.store.arrayIndexExchange(toex, 0);
end

if isFluidOnly == 0
    s = {mag(1).cellMag, mag(2).cellMag, mag(3).cellMag};
    for i = 1:3
        s{i}.arrayIndexExchange(toex, 1);
    end
end

end

function xchgFluidHalos(mass, mom, ener, dir)
s = { mass, ener, mom(1), mom(2), mom(3) };

GIS = GlobalIndexSemantics();

for j = 1:5;
  cudaHaloExchange(s{j}, dir, GIS.topology, s{j}.bcHaloShare);
end

end

