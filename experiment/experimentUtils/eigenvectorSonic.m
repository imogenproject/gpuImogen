function [ev omega] = eigenvectorSonic(rho, csq, v, b, k, direct)
    % This function gives the sonic (hydrodynamic) eigenvector associated with the
    % plane wave with vector k with dispersion relation w = +- c_s |k|
    % direct == 1 choses +, -1 choses - 

    lambda = -sign(direct)*sqrt(csq)*norm(k);

    x = -csq/(lambda*rho);

    % eigenvector: [1, -csq {k} / lambda rho, {0}]
    ev = [1; k(1)*x; k(2)*x; k(3)*x; 0; 0; 0];
    omega = dot(k,v) - lambda;
end
