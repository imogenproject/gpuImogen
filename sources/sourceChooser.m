function S = sourceChooser(run, fluids, mag)
% This function takes the same arguments as the source functions themselves,
% Identifies which source terms are active,
% and chooses an appropriate source function to call at runtime without
% the overhead of this process in a generic source.m file.

if run.geometry.pGeometryType == ENUM.GEOMETRY_CYLINDRICAL; useCyl = 1; else useCyl = 0; end
if run.geometry.frameRotationOmega ~= 0; useRF  = 1; else useRF  = 0; end
if run.potentialField.ACTIVE;    usePhi = 1; else usePhi = 0; end
if numel(run.fluid) > 1;         use2F  = 1; else use2F  = 0; end

% radiation? handle by 2fluid?
%cyl	rf	phi	2f	rad	| SOLUTION		
%0	0	0	0	c	| blank fcn		CHECK
%1	0	0	0	c	| cyl only		CHECK
%0	1	0	0	c	| rot frame only	CHECK
%1	1	0	0	c	| call composite	CHECK
%0	0	1	0	c	| call scalarPot	CHECK
%1	0	1	0	c	| call composite	CHECK
%0	1	1	0	c	| call composite	CHECK
%1	1	1	0	c	| call composite	CHECK
%0	0	0	1	N	| 2f-drag		CHECK
%1	0	0	1	N	| cyl, 2f, cyl		CHECK
%0	1	0	1	N	| rf, 2f, rf		CHECK
%1	1	0	1	N	| 2f, cmp, 2f		CHECK
%0	0	1	1	N	| 2f, phi, 2f		CHECK
%1	0	1	1	N	| 2f, composite, 2f	CHECK
%0	1	1	1	N	| 2f, composite, 2f	CHECK
%1	1	1	1	N	| 2f, composite, 2f	CHECK
%----------------------------------------+-----------------

% We note that two sets of four calls are the same and roll those
% into srcComp and src2f_cmp_2f...

sourcerFunction = useCyl + 2*useRF + 4*usePhi + 8*use2F;
switch sourcerFunction;
    case 0; S  = @srcBlank;
    case 1; S  = @src1000;
    case 2; S  = @src0100;
    case 3; S  = @srcComp;
    case 4; S  = @src0010;
    case 5; S  = @srcComp;
    case 6; S  = @srcComp;
    case 7; S  = @srcComp;

    case 8; S  = @src0001;
    case 9; S  = @src1001;
    case 10; S = @src0101;
    case 11; S = @src2f_cmp_2f;
    case 12; S = @src0011;
    case 13; S = @src2f_cmp_2f;
    case 14; S = @src2f_cmp_2f;
    case 15; S = @src2f_cmp_2f;
end



end
