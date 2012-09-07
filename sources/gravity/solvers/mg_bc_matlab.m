function phiSet = mg_bc_matlab(rhos, poss, bvecSet, coarseConst, h)
% Calculate boundary conditions using multigrid summation
% This function is for generating _BOUNDARY CONDITIONS FOR
% A LINEAR SOLVER_ so it _ASSUMES REGULAR GRID SPACING_
%
%>> rhos         Cell array of mass generated by massQuantization()            cell
%>> poss         Cell array of position generated by massQuantization()        cell
%>> bvec         [9x1] double describing where to sum at                       double
%>> coarseConst  Constant for tuning accuracy vs speed                         double
%<< phi          Computed potential array                                      double

%--- Step one: Pull bvec apart and initialize ---%
%        This is using the 'bvec' argument meant for mg_bc for dropin compatibility for now
%        This pulls it apart and generates a set of points we're looking to generate conditions for

global MGBC_MATLAB_SELFPOT_RAD;

MGBC_MATLAB_SELFPOT_RAD = h/3.836451287;

phiSet = cell(size(bvecSet));

bigXes = [];
bigYes = [];
bigZes = [];
for q = 1:numel(phiSet)
    bvec = bvecSet{q};

    x0 = bvec(1:3);
    xf = max(bvec(1:3)+1, bvec(4:6));
    nsteps = max(bvec(7:9)+1,1)-1;
    nsteps(nsteps == 0) = .1;

    xarr = x0(1):(xf(1)-x0(1))/nsteps(1):xf(1);
    yarr = x0(2):(xf(2)-x0(2))/nsteps(2):xf(2);
    zarr = x0(3):(xf(3)-x0(3))/nsteps(3):xf(3);

    [planeX planeY planeZ] = ndgrid(xarr,yarr,zarr);
    bigXes = [bigXes; reshape(planeX, [numel(planeX) 1])];
    bigYes = [bigYes; reshape(planeY, [numel(planeY) 1])];
    bigZes = [bigZes; reshape(planeZ, [numel(planeZ) 1])];
end

phi = zeros(size(bigXes));

%--- Include the 3rd 1 if necessary ---%
topLevelDims = size(rhos{1}); % Coarsest, should be ~ 4x2x2 or less
if numel(topLevelDims) == 2
    topLevelDims(3)=1;
end

[ctX ctY ctZ] = ndgrid(1:topLevelDims(1), 1:topLevelDims(2), 1:topLevelDims(3));
ctX = reshape(ctX, [numel(ctX) 1]);
ctY = reshape(ctY, [numel(ctY) 1]);
ctZ = reshape(ctZ, [numel(ctZ) 1]);

%rhosDims = ones([numel(rhos) 3]);
%for q = 1:numel(rhos); rhosDims(q,:) = size(rhos{q}); end

%--- Step two: Iterate over the largest blocks ---%
%        WARNING: Float point addition does not commute
%        WARNING: In parallel this causes 12th-decimal uncertainty in the potential
ccprime = coarseConst * 2^(numel(rhos)+1);

parfor n = 1:numel(ctX)
    phi = phi + sumThisBlock(rhos, poss, [ctX(n) ctY(n) ctZ(n)], 1, bigXes, bigYes, bigZes, ccprime);
end

ndone = 0;
for q = 1:numel(phiSet)
    phiSet{q} = phi(1+ndone:prod(bvecSet{q}(7:9)+1)+ndone);
    phiSet{q} = reshape(phiSet{q}, bvecSet{q}(7:9)+1);
    ndone = ndone + prod(bvecSet{q}(7:9)+1);

end

%phi = reshape(phi, bvec(7:9)+1);

end

%______________________________________________________________________
%--- Tree walker routine ---%
function phi = sumThisBlock(rhos, poss, blkId, blkLevel, xes, yes, zes, coarseConst)
% rhos 
% poss     Cell arrays same as before
% blkId    [xindex yindex zindex level] to look at
% xes
% yes
% zes      Set of <x y z> points to accumulate potential at

global MGBC_MATLAB_SELFPOT_RAD;

blkPos = poss{blkLevel}(:,blkId(1),blkId(2),blkId(3));

phi = zeros(size(xes));

%--- Calculate the radii of all given cells from this block and accumulate if far enough ---%
rad = sqrt((xes - blkPos(1)).^2 + (yes - blkPos(2)).^2 + (zes - blkPos(3)).^2);

rad(rad < 1e-4) = MGBC_MATLAB_SELFPOT_RAD;

sel = (rad < coarseConst/2^blkLevel);
phi(~sel) = -rhos{blkLevel}(blkId(1),blkId(2),blkId(3)) ./ rad(~sel);
%phi(sel) = 0;

% FIXME: If this is to be usable for general gravity, we need to account for
% the cell's self potential here. If it's just for BCs, eh.

%--- Prevent unlimited recursion ---%
%        Stop at the maximum level of refinement
%        Add in any cells too close
if blkLevel >= numel(rhos); 
    phi(sel) = -rhos{blkLevel}(blkId(1),blkId(2),blkId(3)) ./ rad(sel);
    return;
end

%--- Stop recursing if there's no cells ---%
if max(sel) == 0; return; end

%--- Identify the next set of blocks to recurse over ---%
%        We may have divided an odd number of cells in half when reducing,
%        So the next step up may have two or only one steps in any of the 3 directions
%        This determines that by looking at whether the # of cells there is even or odd
ns = size(rhos{blkLevel+1});
if numel(ns) == 2; ns(3) = 1; end
%ns = rhosDims(blkLevel+1,:);

nextBlock0 = blkId * 2 - 1;
nextsteps = 1.0*(nextBlock0 < ns);

VV = phi(sel);
alpha = xes(sel);
beta  = yes(sel);
gamma = zes(sel);

for cvx = 0:nextsteps(1);
    for cvy = 0:nextsteps(2);
        for cvz = 0:nextsteps(3);
            VV = VV + sumThisBlock(rhos, poss, nextBlock0+[cvx cvy cvz], blkLevel+1, alpha, beta, gamma, coarseConst);
        end
    end
end

phi(sel) = VV;

end