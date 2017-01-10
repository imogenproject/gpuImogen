function source_alterFrameRotation(tracker, run, mass, ener, mom, newOmega)
% This remaps the momentum density (and associed energy density)
% when the frame's rotation rate is changed from w0 to w1.

if run.geometry.pGeometryType == ENUM.GEOMETRY_SQUARE
    
    % Calculate the newOmega in velocity due to change in rotation rate
    [X, Y, ~] = run.geometry.ndgridSetIJK('pos');
    
    deltaOMega = newOmega - tracker.omega;
    
    X = (X - tracker.rotateCenter(1))*deltaOMega;
    Y = (Y - tracker.rotateCenter(2))*deltaOMega;
    
    % Remember the original kinetic energy density which includes the original w0 term
    T0 = mom(1).array.^2 + mom(2).array.^2;
    % Alter the momentum arrays
    mom(1).array = mom(1).array + mass.array.*Y;
    mom(2).array = mom(2).array - mass.array.*X;
    % Update energy density to reflect changed KE density
    ener.array = ener.array + .5*(mom(1).array.^2+mom(2).array.^2 - T0)./mass.array;
    
    tracker.omega = newOmega;
end

if run.geometry.pGeometryType == ENUM.GEOMETRY_CYLINDRICAL
    % In cylindrical coordinates, ONLY rotation about the coordinate axis is supported
    if mpi_amirank0() && any(tracker.rotateCenter ~= 0)
        disp('NOTE: run.frameParameter.rotateCenter was nonzero while we are using cylindrical coords.');
        disp('NOTE: Cylindrical coordinates rotate about the axis regardless of this parameter.');
    end
    
    [R, ~, ~] = run.geometry.ndgridSetIJK('pos');
    
    deltaOmega = newOmega - tracker.omega;

    % Remember the original kinetic energy density which includes the original w0 term
    T0 = mom(2).array.^2;
    % Alter the momentum arrays
    mom(2).array = mom(2).array - mass.array.*R*deltaOmega;
    % Update energy density to reflect changed KE density
    ener.array = ener.array + .5*(mom(2).array.^2 - T0)./mass.array;
    
    tracker.omega = newOmega;
end

end
