function D = fluidDetailModel(name)
% D = fluidDetailModel(string name) returns the thermodynamic parameters stored
% here associated with the name. These are stored in SI units.
% Structure contents:
% D.gamma: Adiabatic index (gamma is taken as constant in the code)
% D.sigma: Effective hard-sphere geometric cross section evaluated at T=0C
%          (goes to computation of mean free path for gas-dust drag)
%          Given the definition for single-species MFP as
%              lambda = 1/(n pi d_{12}^2),
%          D.sigma equal pi * d_{12}^2
%          where d_{12} = .5*(diameter 1 + diameter 2)
%          i.e., SIGMA IS FOUR TIMES THE HARD SPHERE AREA
% D.radTindex: radius scales as (T / 273K)^radTindex
% D.mass: mass of a particle
% D.dynViscosity: dynamic viscosity in SI units evaluated at 0C
% D.viscTindex: viscosity scales as (T/273K)^viscTindex: >= 0.5
%
% NOTES:
% Sigma is defined by the viscosity as follows:
% Given a measured viscosity, the equations for viscosity of point centers of repulsion define

% conventional mass unit for atoms/molecules
amu = 1.66053904e-27;

% Some known units for things we will be likely interested in
massO2 = 15.999*2*amu;
massN2 = 14.0067*2*amu;
massH2 = 2.016*amu;
massHe = 4.031882*amu;
massArgon = 39.948*amu;

D = struct('gamma',5/3,'sigma',1,'sigmaTindex',0, 'mass',1.66e-27,'dynViscosity',10e-6,'viscTindex',0.5,'minMass',1,'kBolt',1.381e-23);
D.minMass = 0;

if nargin == 0; D = fluidDetailModel('cold_molecular_hydrogen'); return; end

% 100% pure hydrogen below 200K (rotation frozen out)
if strcmp(name,'cold_molecular_hydrogen')
    D.gamma = 5/3; 
    D.mass = massH2;
    D.viscTindex = 0.7; % point center of force: nu = 11
    D.sigmaTindex = 0.2; % 2 / (nu-1)
    D.dynViscosity = 8.9135e-6; % NIST REFPROP database
    D.sigma = generateSigma(298.15, D.mass, D.dynViscosity);
    return;
end

% 100% pure hyrogen from 200~1000K 
if strcmp(name,'warm_molecular_hydrogen')
    D.gamma = 7/5; 
    D.mass = massH2;
    D.viscTindex = 0.7; % point center of force: nu = 11
    D.sigmaTindex = 0.2; % 2 / (nu-1)
    D.dynViscosity = 8.9135e-6; % NIST REFPROP database
    D.sigma = generateSigma(298.15, D.mass, D.dynViscosity); 
    return;
end

% Pure hydrogen from 1000-3000K
if strcmp(name,'hot_molecular_hydrogen')
    D.gamma = 9/7;
    D.mass = massH2;
    D.sigma = 2.623e-19;
    D.sigmaTindex = 0.2;
    D.dynViscosity = 0;
    D.viscTindex = 0.7;
    return;
end

% These three entries correspond a mix of 75% molecular H2 and 25% He by mass.
% This corresponds to mole & number fractions of 86% H2 and 14% He.
% The rounded H2/He number fractions, 0.86 and 0.14, are used for all calculations of
% parameters of the mixture below.
% Note that the Imogen equation of state ignores the anomalous specific heat of Hydrogen at
% low temperatures caused by the ortho/para splitting

% Mixture of 75% H2 and 25% He by mass at low temperature (<~200K): rovibe both frozen out
if strcmp(name, 'cold_h2he_cosmic')
    D.gamma = 5/3;
    D.mass = 3.3458e-27;
    D.sigma = 2.623e-19;
    D.sigmaTindex = 0;
    D.dynViscosity = 0;
    D.viscTindex = 0;
    return;
end

% Cosmic H2/He mix at warmer temps (200-1000K): rotation available
if strcmp(name, 'warm_h2he_cosmic')
    D.gamma = 1.424; 
    D.mass = 3.3458e-27;
    D.sigma = 2.623e-19;
    D.sigmaTindex = 0;
    D.dynViscosity = 0;
    D.viscTindex = 0;
    return;
end

% Cosmic H2/He mix at high temps (1000-3000K): vibration available
if strcmp(name, 'hot_h2he_cosmic')
    D.gamma = 1.311; 
    D.mass = 3.3458e-27;
    D.sigma = 2.623e-19;
    D.sigmaTindex = 0;
    D.dynViscosity = 0;
    D.viscTindex = 0;
    return;
end

% Air is by mass 78% N2, 21% O2, 1% Ar to an excellent approximation
% The corresponding number fractions are
% .80341 N2, .18936 O2, .00723 Ar (rounded to sum to exactly 1) 
if strcmp(name,'cold_air')
    D.gamma = 5/3;
    D.mass = 28.9695*amu;
    D.dynViscosity = 17.15e-6;
    D.viscTindex = 0.7681;
    D.sigma = generateSigma(298.15, D.mass, D.dynViscosity);
    D.sigmaTindex = .2681;
    return;
end

if strcmp(name,'warm_air')
    D.gamma = 7/5;
    D.mass = 28.9695*amu;
    D.dynViscosity = 18.4918e-6;
    D.viscTindex = 0.7681;
    D.sigma = generateSigma(298.15, D.mass, D.dynViscosity);
    D.sigmaTindex = .2681;
    return;
end

% Thermo data for pure helium gas
if strcmp(name, 'helium')
    D.gamma = 5/3;
    D.mass = 4.031882*amu;
    D.dynViscosity = 1.8743e-5;
    D.viscTindex = .6527;
    D.sigma = 1.5040e-19
    D.sigma = generateSigma(298.15, D.mass, D.dynViscosity)
    D.sigmaTindex = .1527;
    return;
end

if strcmp(name,'10um_iron_balls')
    D.gamma = 1.01; D.sigma = pi*25e-12; D.mass = 3e-11;
    return;
end

% Default model = cold molecular hydrogen
D = fluidDetailModel('cold_molecular_hydrogen');

end

function s = generateSigma(Tref, mass, visc)
% Utility line computes the exact single-species scattering cross section required to
% merge correctly to Epstein regime at large Kn

    %s = sqrt(2/27) * 1.7 * sqrt(1.381e-23 * Tref * mass) / visc;
    s = sqrt(2/27) * 1.7 * sqrt(1.381e-23 * Tref * mass) / visc;
end
