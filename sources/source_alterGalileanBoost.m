function source_alterGalileanBoost(run, mass, ener, mom, newVelocity)
% This remaps the momentum density (and associed energy density)
% when the frame's rotation rate is changed from w0 to w1.
    jump = newVelocity - run.frameTracker.centerVelocity;

    % Remember the original kinetic energy density
    T0 = mom(1).array.^2 + mom(2).array.^2 + mom(3).array.^2;

    % Alter the momentum arrays
    mom(1).array = mom(1).array + mass.array*jump(1);
    mom(2).array = mom(2).array + mass.array*jump(2);
    mom(3).array = mom(3).array + mass.array*jump(3);

    % Update energy density to reflect changed KE density
    ener.array = ener.array + .5*((mom(1).array.^2+mom(2).array.^2 + mom(3).array.^2) - T0)./mass.array;

    run.frameVelocity = newVelocity;
end
