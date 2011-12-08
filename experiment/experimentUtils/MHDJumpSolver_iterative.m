function result = MHDJumpSolver_iterative(ms, ma, theta, GAMMA)
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
    %            pre: pr, rho, ms, ma, vz=0, bz=0, tang=tan(theta) gamma=5/3
    %           post: BX = bx (divergance eqn), VZ=0, BZ=0
    %   Uknowns:
    %            pre: vx, vj, bx, bj
    %           post: RHO, VX, VJ, PR, BJ 
    
    % Translate these into names used in Mathematica formulation

    theta = theta * pi / 180; 

    vxpre = ms * sqrt(GAMMA)
    vypre = vxpre * tan(theta);
    
    bx = vxpre / ma;
    bypre = bx * tan(theta);
    
    rhopre = 1;
    Ppre = 1;
    g = GAMMA;
    tt = tan(theta);

    % Calculate the Hydrodynamic shock solution as our initial guess to assure the iteration
    % converges to the shock solution, not the comoving solution

    vxpost = (2* g* Ppre - rhopre* vxpre^2 + g *rhopre* vxpre^2)/((1 + g)* rhopre* vxpre)
vxpost = 3.4043


 


%    a4 = (1-3*g)*pxpre^3;
%    a3 = pxpre^2*(bx^2*(6*g-2) + g*(bypre^2+2*Ppre + 2*txpre));
%    a2 = pxpre*(bx^4*(1-3*gh + txpre*(2*bypre^2*(g-1)+2*g*Ppre +(g-1)*txpre)+bx^2*(bypre^2*(1-3*g)-4*g*(Ppre+txpre)  ));


%    a1 = bypre^2*(g-2)*txpre^2 + 2*bx^4*g*(Ppre + txpre) - 2*bx^2*txpre*(bypre^2*(g-2) + 2*g*Ppre +(g-1)*txpre) + 4*bx^3*bypre*(g-1)*pxpre*vypre - 4*bx*bypre*(g-1)*rhopre^2*vxpre^3*vypre;
%    a1 = bypre^2*(g-2)* rhopre^2* vxpre^4 +  2* bx^4* g* (Ppre + rhopre* vxpre^2) -  2* bx^2* rhopre* vxpre^2 *(bypre^2 *(-2 + g) + 2 *g* Ppre - rhopre* vxpre^2 +     g* rhopre* vxpre^2) + 4* bx^3* bypre* (-1 + g) *rhopre* vxpre *vypre -  4 *bx* bypre* (-1 + g)* rhopre^2* vxpre^3* vypre;

 %   a0 = bx^2*(-3*bypre^2*(g-1)*txpre*vxpre + bx^2*vxpre*(4*bypre^2*(g-1) + 2*g*Ppre +(g-1)*txpre) - 4*bx^3*bypre*(g-1)*vypre + 4*bx*bypre*(g-1)*txpre*vypre);

    % Solve for postshock vx: Use quartic solver
    %a4 = ((rhopre^3 *vxpre^3 + g *rhopre^3 *vxpre^3));
    %a3 = ((-2 *bx^2 *rhopre^2 *vxpre^2 - 2 *bx^2 *g *rhopre^2 *vxpre^2 - 2 *g *Ppre *rhopre^2 *vxpre^2 - 2 *bx^2 *g *rhopre^2 *tt^2 *vxpre^2 - 2 *g *rhopre^3 *vxpre^4));
    %a2 = ((bx^4 *g *rhopre* vxpre + 4 *bx^2 *g *Ppre *rhopre *vxpre + 3 *bx^4 *g *rhopre *tt^2 *vxpre + bx^4 *rhopre *(1 + tt^2)* vxpre + 4* bx^2* g *rhopre^2 *vxpre^3 + 2 *g *Ppre* rhopre^2 *vxpre^3 - 2 *bx^2 *rhopre^2 *tt^2 *vxpre^3 + 2 *bx^2 *g *rhopre^2 *tt^2* vxpre^3 - rhopre^3 *vxpre^5 + g* rhopre^3 *vxpre^5));
    %a1 = ((-2 *bx^4 *g *Ppre - 2 *bx^4* g *rhopre *vxpre^2 - 4 *bx^2* g *Ppre* rhopre* vxpre^2 - 4* bx^4 *g *rhopre *tt^2 *vxpre^2 + 2 *bx^2* rhopre^2* vxpre^4 - 2 *bx^2 *g *rhopre^2 *vxpre^4 + 2 *bx^2 *rhopre^2 *tt^2 *vxpre^4));
    %a0 = ((2 *bx^4 *g *Ppre *vxpre + bx^4 *g *rhopre *vxpre^3 + bx^4 *g *rhopre *tt^2* vxpre^3 - bx^4 *rhopre* (1 + tt^2) *vxpre^3));
    
%[a0 a1 a2 a3 a4]

%    vpost = solveQuartic(a4, a3, a2, a1, a0)
%vpost = roots([a4 a3 a2 a1 a0])
%    imag(vpost)

%    vpost = real(vpost(imag(vpost) < 1e-11)); % The MHD equations are assholes and the quartic solver is an asshole so this is necessary.
%    vxpost = min(vpost); % The lesser is the one containing a discontinuity; The other corresponds to no jump.

    bypost = (-bx^2*bypre + bypre*rhopre*vxpre^2)/(rhopre*vxpost*vxpre - bx^2);
    vypost = (bx*bypost - bx*bypre + rhopre*vxpre*vypre) / (rhopre*vxpre);
    Ppost = -bypost^2 + bypre^2 + Ppre - rhopre *vxpost *vxpre + rhopre *vxpre^2;
    rhopost = rhopre *vxpre / vxpost;

    result.mass       = [1; rhopost];
    result.pressure   = [1; Ppost];
    result.velocity   = [vxpre vxpost; vypre vypost; 0 0;];
    result.magnet     = [bx bx; bypre bypost; 0 0];
    result.theta      = theta;
    result.sonicMach  = ms;
    result.alfvenMach = ma;

evalF(1, vxpre, vypre, bx, bypre, 1, rhopost, vxpost, vypost, bypost, Ppost, GAMMA)

qpost = [rhopost, vxpost, vypost, bypost, Ppost]';

for iter = 1:10
   J = getJ(1, vxpre, vypre, bx, bypre, 1, qpost(1), qpost(2), qpost(3), qpost(4), qpost(5), GAMMA);
   qpost = qpost - .1*J^-1 * evalF(1, vxpre, vypre, bx, bypre, 1, qpost(1), qpost(2), qpost(3), qpost(4), qpost(5), GAMMA);
end

qpost

rhopost = qpost(1);
vxpost = qpost(2);
vypost = qpost(3);
bypost = qpost(4);
Ppost = qpost(5);

evalF(1, vxpre, vypre, bx, bypre, 1, rhopost, vxpost, vypost, bypost, Ppost, GAMMA)

end

function f = evalF(rho1, vx1, vy1, bx, by1, P1, rho2, vx2, vy2, by2, P2, g)

f(1) = vx2*by2 - vy2*bx - vx1*by1 + vy1*bx;
f(2) = rho2*vx2 - rho1*vx1;
f(3) = rho2*vx2*vx2 + P2 + by2*by2/2 - rho1*vx1*vx1 - P1 - by1*by1/2;
f(4) = rho2*vx2*vy2 - bx*by2 - rho1*vx1*vy1 + bx*by1;
f(5) = .5*rho2*(vx2^2+vy2^2)*vx2 + g*P2*vx2/(g-1) + by2*(vx2*by2-bx*vy2) -  .5*rho1*(vx1^2+vy1^2)*vx1 - g*P1*vx1/(g-1) - by1*(vx1*by1-bx*vy1);

f=f';

end

function J = getJ(rho1, vx1, vy1, bx, by1, P1, rho2, vx2, vy2, by2, P2, g)

J(1,1) = 0;
J(2,1) = vx2;
J(3,1) = vx2^2;
J(4,1) = vx2*vy2;
J(5,1) = .5*(vx2^2+vy2^2)*vx2;

J(1,2) = by2;
J(2,2) = rho2;
J(3,2) = 2*rho2*vx2;
J(4,2) = rho2*vy2;
J(5,2) = 1.5*rho2*vx2^2 + .5*rho2*vy2^2 + g*P2/(g-1) + by2^2;


J(1,3) = -bx;
J(2,3) = 0;
J(3,3) = 0;
J(4,3) = rho2*vx2;
J(5,3) = rho2*vx2*vy2 - by2*bx;

J(1,4) = vx2;
J(2,4) = 0;
J(3,4) = by2;
J(4,4) = -bx;
J(5,2) = 2*by2*vx2;

J(1,5) = 0;
J(2,5) = 0;
J(3,5) = 1;
J(4,5) = 0;
J(5,5) = g*vx2/(g-1);
end


