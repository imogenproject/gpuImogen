function source_alterFrameRotation(run, fluids, newOmega)
% This remaps the momentum density (and associed energy density)
% when the frame's rotation rate is changed from w0 to w1.

if run.geometry.pGeometryType == ENUM.GEOMETRY_SQUARE
    
    % Calculate the newOmega in velocity due to change in rotation rate
    [X, Y, ~] = run.geometry.ndgridSetIJK('pos');
    
    deltaOmega = newOmega - run.geometry.frameRotationOmega;
    
    X = (X - run.geometry.frameRotationCenter(1))*deltaOmega;
    Y = (Y - run.geometry.frameRotationCenter(2))*deltaOmega;
    
    for x=1:numel(fluids)
        % Remember the original kinetic energy density which includes the original w0 term
        T0 = fluids(x).mom(1).array.^2 + fluids(x).mom(2).array.^2;
        % Alter the momentum arrays
        fluids(x).mom(1).array_NewBC(fluids(x).mom(1).array + fluids(x).mass.array.*Y);
        fluids(x).mom(2).array_NewBC(fluids(x).mom(2).array - fluids(x).mass.array.*X);
        % Update energy density to reflect changed KE density
        fluids(x).ener.array_NewBC(fluids(x).ener.array ...
            + .5*(fluids(x).mom(1).array.^2+fluids(x).mom(2).array.^2 - T0)./fluids(x).mass.array);
    end
    run.geometry.frameOmega = newOmega;
end

if run.geometry.pGeometryType == ENUM.GEOMETRY_CYLINDRICAL
    % In cylindrical coordinates, ONLY rotation about the coordinate axis is supported
    if mpi_amirank0() && any(run.geometry.frameRotationCenter ~= 0)
        disp('NOTE: run.frameParameter.rotateCenter was nonzero while we are using cylindrical coords.');
        disp('NOTE: Cylindrical coordinates rotate about the axis regardless of this parameter.');
    end
    
    [R, ~, ~] = run.geometry.ndgridSetIJK('pos');
    
    deltaOmega = newOmega - run.geometry.frameRotationOmega;

    for x=1:numel(fluids)
    % Remember the original kinetic energy density which includes the original w0 term
    T0 = fluids(x).mom(2).array.^2;
    % Alter the momentum arrays
    fluids(x).mom(2).array_NewBC(fluids(x).mom(2).array - fluids(x).mass.array.*R*deltaOmega)
    % Update energy density to reflect changed KE density
    fluids(x).ener.array_NewBC(fluids(x).ener.array + .5*(fluids(x).mom(2).array.^2 - T0)./fluids(x).mass.array);
    end
    
    run.geometry.frameRotationOmega = newOmega;
end

end
