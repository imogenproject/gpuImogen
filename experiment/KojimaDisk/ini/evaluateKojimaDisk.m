function [rho, moma, momb, eint] = evaluateKojimaDisk(q, gamma, radiusRatio, starMG, minRhoCoeff, rPoints, phiPoints, zPoints, geomode)
% Return the density & momentum density of an equilibrium NSG rotating disk evaluated
% at the given r and z points.
%>> q               angular velocity exponent; omega = omega_0 * r^-q.                  double
%>> gamma           polytropic index used for equation of state.                        double
%>> radiusRatio     inner radius / radius of density max.                               double
%>> starMG          specified mass of central star for normalization.                   double
%>> minRhoCoeff     What fraction of the density max to assign to zero-mass regions     double
%>> rPoints         radii to evaluate at
%>> phiPoints       phi coordinates to evaluate at
%>> zPoints         Z points to evaluate at 
%>> geomode         If ENUM.GEOMETRY_CYLINDRICAL 

%Return values:
%<< mass            equhilibrium mass density.                                           double(GRID)
%<< mom             equilibrium y momentum density.                                     double(GRID)

    diskInfo           = kojimaDiskParams(q, radiusRatio, gamma);

    xq                 = 2 - 2*q; %Useful constant
    starDistance       = sqrt(rPoints.^2 + zPoints.^2);

    %--- Constants of integration ---%
    %       hsq is useless here because of our unit scaling.
    %       c1 is the integration constant from solving the Bernoulli equation.
    hsq                = xq * (1/diskInfo.rout - 1/diskInfo.rin)/(diskInfo.rin^xq - diskInfo.rout^xq);
    c1                 = -1/diskInfo.rin - (hsq/xq)*(diskInfo.rin^xq);

    lomass             = minRhoCoeff * diskInfo.rhomax;

    %--- Calculates the pressure integral (Bernoulli equation), clamps #s < 0, solves for rho ---%
    bernoulli          = c1 + (1 ./ starDistance) + (hsq / xq) * rPoints.^xq;
    isPartOfDisk       = (bernoulli > 0);
    bernoulli(~isPartOfDisk) = 0;

    rho                = max((bernoulli * (gamma-1)/gamma).^(1/(gamma-1)), lomass);
    rho(isPartOfDisk)  = max(rho(isPartOfDisk), lomass * ENUM.GRAV_FEELGRAV_COEFF);

    eint = zeros(size(rho));
    eint(isPartOfDisk) = rho(isPartOfDisk).^gamma / (gamma-1);
    
    mom                = rPoints.^(1-q);

    %--- Now begin the numeric fudging ---%

    % Set an external soundspeed for finite temperature: .01 yields T = 1e-4... ok
    cextern=.01;
    
    if 0 % This embeds an isothermal Keplerian disk outside the Kojima disk in an attempt
         % to reduce the startup transient. It doesn't work very well.
        rbdy = 1.01*diskInfo.rin;
        rhobdy=  c1 + (1 ./ rbdy) + (hsq / xq) * rbdy^xq;
        rhobdy = (rhobdy*(gamma-1)/gamma)^(1/(gamma-1));
    
        efact = -hsq./rPoints + 1./starDistance + hsq/rbdy - 1/rbdy;
        efact = efact / cextern^2;

        outside = (rho == lomass);

        mom(outside) = rPoints(outside).^-.5;

        rho(outside) = rhobdy*exp(efact(outside));
        rho(rho < lomass) = lomass; % prevent *excessive* vacuumeyness

        eint(outside) = rho(outside) * cextern^2 / (gamma*(gamma-1));
    end

    % Make sure velocity is zero where it ought be zero
    outside = (rho == lomass);
    mom(outside) = 0;
   
    % Convert to momentum density
    mom = mom .* rho;
    
    % Switch momentum off where gravity won't act
    % FIXME: this and ENUM.GRAV_FEELGRAV_COEFF do not sync with sources/source.m
    mom(rho < 4*lomass) = 0;
    
    if 0 % Another bad attempt to reduce the inner edge transient (by making it just fall in)
         % This, too, does not work very well.
        dr = rPoints(2)-rPoints(1);
        mu = 2; nu = 2;

        bz = (rPoints - radiusRatio > -mu*dr) & (rPoints - radiusRatio < (nu-mu)*dr);

        mom(bz) = rPoints(bz).^(1-q) .* rho(bz) .* (rPoints(bz) - radiusRatio + mu*dr) / (nu*dr); 
    end

    % momentum is projected into x/y or r/theta space. mom_z is always zero.
    if geomode == ENUM.GEOMETRY_CYLINDRICAL
        momb = mom;
        moma = zeros(size(mom));
    elseif geomode == ENUM.GEOMETRY_SQUARE
        moma = -mom .* sin(phiPoints);
        momb = mom .* cos(PhiPoints);
    end

end

