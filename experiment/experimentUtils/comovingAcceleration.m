function A = comovingAcceleration(fluid, gravPot, geometry)

P = fluid.calcPressureOnCPU();
rho = fluid.mass.array;

if geometry.pGeometryType == ENUM.GEOMETRY_CYLINDRICAL
    [radius, ~, ~] = geometry.ndgridSetIJK('pos');
    
    if geometry.globalDomainRez(3) > 1
        [P1, P2, P3] = gradient(P, geometry.d3h(1), geometry.d3h(2), geometry.d3h(3));
        [G1, G2, G3] = gradient(gravPot, geometry.d3h(1), geometry.d3h(2), geometry.d3h(3));
    else
        [P1, P2] = gradient(P, geometry.d3h(1), geometry.d3h(2));
        [G1, G2] = gradient(gravPot, geometry.d3h(1), geometry.d3h(2));
        P3 = 0; G3 = 0;
    end
    
    P2 = P2 ./ radius;
    G2 = G2 ./ radius;
    
    A = { P1 ./ rho + G1, P2 ./ rho + G2, P3 ./ rho + G3};
    
    vphi = fluid.mom(2).array ./ rho;
    if geometry.frameRotationOmega ~= 0
        vphi = vphi + radius * geometry.frameRotationOmega;
    end
    
    A{2} = A{2} + vphi.^2 ./ radius;
else
    if size(P,3) > 1
        [P1, P2, P3] = gradient(P, geometry.d3h(1), geometry.d3h(2), geometry.d3h(3));
        if numel(gravPot) > 1
            [G1, G2, G3] = gradient(gravPot, geometry.d3h(1), geometry.d3h(2), geometry.d3h(3));
        else
            G1 = 0; G2 = 0; G3 = 0;
        end
    else
        [P1, P2] = gradient(P, geometry.d3h(1), geometry.d3h(2));
        if numel(gravPot) > 1
            [G1, G2] = gradient(gravPot, geometry.d3h(1), geometry.d3h(2));
        else
            G1 = 0; G2 = 0;
        end
        P3 = 0; G3 = 0;
    end
    
    A = { -P1 ./ rho - G1, -P2 ./ rho + -G2, -P3 ./ rho - G3};
end
 
% FIXME: broken: This will fail for calculating theta direction gradient because h[2] == dtheta, need 1/r...





end