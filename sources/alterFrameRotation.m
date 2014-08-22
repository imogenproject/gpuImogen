function alterFrameRotation(run, mass, ener, mom, newOmega)
% This remaps the momentum density (and associed energy density)
% when the frame's rotation rate is changed from w0 to w1.
	jump = newOmega - run.frameRotateOmega;

	% Calculate the jump in velocity due to change in rotation rate
	GIS = GlobalIndexSemantics();
	[X Y] = GIS.ndgridSetXYZ();
	X = (X - run.frameRotateCenter(1))*jump*run.DGRID{1};
	Y = (Y - run.frameRotateCenter(2))*jump*run.DGRID{2};

	% Remember the original kinetic energy density
	T0 = mom(1).array.^2 + mom(2).array.^2;

	% Alter the momentum arrays
	mom(1).array = mom(1).array + mass.array.*Y;
	mom(2).array = mom(2).array - mass.array.*X;

	% Update energy density to reflect changed KE density
	ener.array = ener.array + .5*(mom(1).array.^2+mom(2).array.^2 - T0)./mass.array;

	run.frameRotateOmega = newOmega;
end
