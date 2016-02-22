function source_alterFrameRotation(tracker, mass, ener, mom, newOmega)
% This remaps the momentum density (and associed energy density)
% when the frame's rotation rate is changed from w0 to w1.

	% Calculate the newOmega in velocity due to change in rotation rate
	GIS = GlobalIndexSemantics();
	[X Y] = GIS.ndgridSetXYZ();

	jump = newOmega - tracker.omega;

	X = (X - tracker.rotateCenter(1))*jump*run.DGRID{1};
	Y = (Y - tracker.rotateCenter(2))*jump*run.DGRID{2};

	% Remember the original kinetic energy density which includes the original w0 term
	T0 = mom(1).array.^2 + mom(2).array.^2;

	% Alter the momentum arrays
	mom(1).array = mom(1).array + mass.array.*Y;
	mom(2).array = mom(2).array - mass.array.*X;

	% Update energy density to reflect changed KE density
	ener.array = ener.array + .5*(mom(1).array.^2+mom(2).array.^2 - T0)./mass.array;

        tracker.omega = newOmega;
end
