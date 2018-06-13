function A = comovingAcceleration(fluid, gravPot, geometry)

P = fluid.calcPressureOnCPU();
rho = fluid.mass.array;

gravon = (rho > 4*fluid.MINMASS);
% Compute v_phi^2 / r
if geometry.pGeometryType == ENUM.GEOMETRY_CYLINDRICAL
    [radius, angle, zee] = geometry.ndgridSetIJK('pos');
    
    [P1, P2, P3] = gradient(P, geometry.d3h(1), geometry.d3h(2), geometry.d3h(3));
    [G1, G2, G3] = gradient(gravPot, geometry.d3h(1), geometry.d3h(2), geometry.d3h(3));
    
    P2 = P2 ./ radius;
    G2 = G2 ./ radius;
    
    A = { P1 ./ rho + G1.*gravon, P2 ./ rho + G2.*gravon, P3 ./ rho + G3.*gravon};
    
    vphi = fluid.mom(2).array ./ rho;
    if geometry.frameRotationOmega ~= 0
        vphi = vphi + radius * geometry.frameRotationOmega;
    end
    
    A{2} = A{2} + vphi.^2 ./ radius;
else
    [P1, P2, P3] = gradient(P, geometry.d3h(1), geometry.d3h(2), geometry.d3h(3));
    [G1, G2, G3] = gradient(gravPot, geometry.d3h(1), geometry.d3h(2), geometry.d3h(3));
    
    A = { P1 ./ rho + G1.*gravon, P2 ./ rho + G2.*gravon, P3 ./ rho + G3.*gravon};
end
 
% FIXME: broken: This will fail for calculating theta direction gradient because h[2] == dtheta, need 1/r...





end