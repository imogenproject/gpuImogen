function flux(run, mass, mom, ener, mag, grav, order)
% This function manages the fluxing routines for the split code by managing the appropriate fluxing 
% order to average out any biasing caused by the Strang splitting.
%
%>< run         run manager object                                                      ImogenManager
%>< mass        mass density                                                            FluidArray
%>< mom         momentum density                                                        FluidArray(3)
%>< ener        energy density                                                          FluidArray
%>< mag         magnetic field                                                          MagnetArray(3)
%>< grav        gravitational potential                                                 GravityArray
%>> order       direction of flux sweep (1 forward/-1 backward)                         int     +/-1

    
    %-----------------------------------------------------------------------------------------------
    % Set flux direction and magnetic index components
    %-------------------------------------------------    
    switch (order)
        case 1;
            directVec       = [1; 2; 3];
            magneticIndices = [2 3; 1 3; 1 2];
        case -1;
            directVec       = [3; 2; 1];
            magneticIndices = [2 1; 3 1; 3 2];
                otherwise;
            run.save.logPrint('%g is not a recognized direction. Fluxing aborted.\n', order);
            return;
    end
    
    directVec       = circshift(directVec,       [mod(run.time.iteration-1,3), 0]);
    magneticIndices = circshift(magneticIndices, [mod(run.time.iteration-1,3), 0]);
    %===============================================================================================
        if (order > 0) %                             FORWARD FLUXING
    %===============================================================================================
                for n=1:3
            if (mass.gridSize(directVec(n)) < 3), continue; end
                        run.parallel.redistributeArrays(directVec(n));
            
                        if run.fluid.ACTIVE
                        xchgIndices(mass, mom, ener, mag, grav, directVec(n));
                        relaxingFluid(run, mass, mom, ener, mag, grav, directVec(n));
                        xchgIndices(mass, mom, ener, mag, grav, directVec(n));
                        end
                        if run.magnet.ACTIVE
                        magnetFlux(run, mass, mom, mag, directVec(n), magneticIndices(n,:));
                        end
                end
    %===============================================================================================        
        else %                                       BACKWARD FLUXING
    %===============================================================================================
                for n=1:3
            if (mass.gridSize(directVec(n)) < 3), continue; end
                        run.parallel.redistributeArrays(directVec(n));
                        
                        if run.magnet.ACTIVE
                        magnetFlux(run, mass, mom, mag, directVec(n), magneticIndices(n,:));
                        end
                        if run.fluid.ACTIVE
                        xchgIndices(mass, mom, ener, mag, grav, directVec(n));
                        relaxingFluid(run, mass, mom, ener, mag, grav, directVec(n));
                        xchgIndices(mass, mom, ener, mag, grav, directVec(n));
                        end
                end
    end
end

function xchgIndices(mass, mom, ener, mag, grav, toex)
l = [1 2 3];
l(1)=toex; l(toex)=1;

s = { mass, ener, mom(1), mom(2), mom(3) };

for i = 1:5
    s{i}.arrayIndexExchange(toex, 1);
    s{i}.store.arrayIndexExchange(toex, 0);
end

s = {mag(1).cellMag, mag(2).cellMag, mag(3).cellMag};
for i = 1:3
    s{i}.arrayIndexExchange(toex, 1);
%    s{i}.store.arrayIndexExchange(toex, 0);
end


end
