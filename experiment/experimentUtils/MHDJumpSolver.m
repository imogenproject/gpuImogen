function [result allvx] = MHDJumpSolver(ms, ma, theta, GAMMA)
% Solves the 3D MHD Jump Conditions for an equilibrium shock wave assuming that the pre-shock region
% conforms to zero z components for both velocity and magnetic field (this
% can always be acheived by a frame transform) as well as unity values for 
% mass density and pressure in the pre-shock region. Gamma has also been set to 5/3 in the equations to 
% simplify the numeric complexity, but that could be changed if necessary. Once the solution of the
% system has been found, the solver looks through the numerical results structure, determines which
% solution is the physical one and returns the data structure for that solution. It will also save
% a mat file with the solution for subsequent loading in the Corrugation shock initializer.
%
%><  run      Manager object for current solver            LinogenManager object
%><  linVars  Prototype data class for solver              LinogenVariables object

    %--- Define jump conditions ---%
    %   Knowns:
    %            pre: P == 1, rho == 1, ms, ma, polytropic gamma, vz=0, bz=0
    %           post: BX = bx (divergence eqn), VZ=0, BZ=0 (rejecting rotational solutions)
    %   Uknowns:
    %            pre: vx, vy, bx, by
    %           post: RHO, VX, VY, P, BY 

    % Solve the preshock flow in terms of the 3 free quantities
    vxpre = ms * sqrt(GAMMA);
    vypre = vxpre * tand(theta);
    
    bx = vxpre / ma;
    bypre = bx * tand(theta);
    
    rhopre = 1;
    Ppre = 1;
    g = GAMMA;
  
    pxpre = rhopre*vxpre;
    txpre = rhopre*vxpre^2;

    % The following equations could be rewritten in terms of the 3 free quantities only (since they
    % completely determine the flow) but writing in terms of physical quantities gives some idea where
    % the terms come from. And I don't care to rewrite them, there's that too.

    % These define a quartic equation sum(vxpost^n a_n) = 0 for vpost
    a4 = (1+g)*pxpre^3;
    a3 = -pxpre^2*(2*bx^2*(g+1) + g*(bypre^2 + 2*Ppre + 2*txpre));
    a2 = pxpre*(bx^4*(g+1) + txpre*(2*bypre^2*(g-1) + 2*g*Ppre +(g-1)*txpre) + bx^2*(bypre^2*(g+1) + 4*g*(Ppre+txpre)));
    a1 = -bypre^2*(g-2)*txpre^2 - 2*bx^4*g*(Ppre+txpre) - 2*bx^2*txpre*(bypre^2*g + 2*g*Ppre +(g-1)*txpre);
    a0 = bx^2*vxpre*(bypre^2*(g-1)*txpre + bx^2*(2*g*Ppre +(g-1)*txpre));
    vpost = solveQuartic(a4, a3, a2, a1, a0);

    % This prevents a confirmed to exist a numerical instability in the solver wherein the 
    % real-valued solutions acquire an O(epsilon) imaginary part due to truncation error and
    % also ejects the nonphysical complex conjugate solutions that arise.
    vpost = real(vpost(abs(imag(vpost)) < 1e-11)); 

    vxpost = min(vpost); % The lesser is the one containing a discontinuity
    bypost = (-bx^2*bypre + bypre*rhopre*vxpre^2)/(rhopre*vxpost*vxpre - bx^2);
    vypost = (bx*bypost - bx*bypre + rhopre*vxpre*vypre) / (rhopre*vxpre);
    Ppost = .5*(-bypost^2 + bypre^2 + 2*Ppre - 2*rhopre*vxpost*vxpre + 2*rhopre*vxpre^2);
    rhopost = rhopre *vxpre / vxpost;

    tpre  = rhopre*(vxpre^2+vypre^2)/2;
    tpost = rhopost*(vxpost^2+vypost^2)/2;
    bsqpre = bx^2+bypre^2;
    bsqpost = bx^2+bypost^2;

    % Now package it up into a nice result struct
    result.mass       = [1; rhopost];
    result.velocity   = [vxpre vxpost; vypre vypost; 0 0;];
    result.magnet     = [bx bx; bypre bypost; 0 0];
    result.pressure   = [1; Ppost];
    result.Etot       = [Ppre/(g-1) + tpre + .5*bsqpre, Ppost/(g-1) + tpost + .5*bsqpost];
    result.theta      = theta;
    result.sonicMach  = ms;
    result.alfvenMach = ma;

    % Of course we should test that this is actually obeying the RH conditions!
    err = evalRankineHugoniotConditions(rhopre, vxpre, vypre, bx, bypre, Ppre, rhopost, vxpost, vypost, bypost, Ppost, GAMMA);
    if norm(err) > 1e-8; fprintf('WARNIN: Norm for obediance of Rankine-Hugoniot equations (lower=better) %g > 1e-8.\n', norm(err)); end

end

% Directly evaluate the Rankine-Hugoniot conditions on either side of the calculated solution.
function f = evalRankineHugoniotConditions(rho1, vx1, vy1, bx, by1, P1, rho2, vx2, vy2, by2, P2, g)
    f(1,1) = vx2*by2 - vy2*bx - vx1*by1 + vy1*bx;
    f(1,2) = rho2*vx2 - rho1*vx1;
    f(1,3) = rho2*vx2*vx2 + P2 + by2*by2/2 - rho1*vx1*vx1 - P1 - by1*by1/2;
    f(1,4) = rho2*vx2*vy2 - bx*by2 - rho1*vx1*vy1 + bx*by1;
    f(1,5) = .5*rho2*(vx2^2+vy2^2)*vx2 + g*P2*vx2/(g-1) + by2*(vx2*by2-bx*vy2) -  .5*rho1*(vx1^2+vy1^2)*vx1 - g*P1*vx1/(g-1) - by1*(vx1*by1-bx*vy1);
end

