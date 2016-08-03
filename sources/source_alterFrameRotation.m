function source_alterFrameRotation(tracker, run, mass, ener, mom, newOmega)
% This remaps the momentum density (and associed energy density)
% when the frame's rotation rate is changed from w0 to w1.

	% Calculate the newOmega in velocity due to change in rotation rate
	[X, Y, ~] = run.geometry.ndgridSetIJK('pos');

	jump = newOmega - tracker.omega;

	X = (X - tracker.rotateCenter(1))*jump;
	Y = (Y - tracker.rotateCenter(2))*jump;

	% Remember the original kinetic energy density which includes the original w0 term
	T0 = mom(1).array.^2 + mom(2).array.^2;

	% Alter the momentum arrays
	mom(1).array = mom(1).array + mass.array.*Y;
	mom(2).array = mom(2).array - mass.array.*X;

	% Update energy density to reflect changed KE density
	ener.array = ener.array + .5*(mom(1).array.^2+mom(2).array.^2 - T0)./mass.array;
    
    tracker.omega = newOmega;
end
