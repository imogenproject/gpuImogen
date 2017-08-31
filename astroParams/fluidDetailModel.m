function D = fluidDetailModel(name)
% D = fluidDetailModel(string name) returns the thermodynamic parameters stored
% here associated with the name. These are stored in SI units.
D = struct('gamma',1,'sigma',1, 'mu',1,'minMass',1);
D.minMass = 0;

if nargin == 0; D = fluidDetailModel('cold_molecular_hydrogen'); return; end

if strcmp(name,'cold_molecular_hydrogen')
    D.gamma = 5/3; D.sigma = 2.623e-19; D.mu = 3.3458e-27;
    return;
end

if strcmp(name,'warm_molecular_hydrogen')
    D.gamma = 7/5; D.sigma = 2.623e-19; D.mu = 3.3458e-27;
    return;
end

if strcmp(name,'cold_air')
    D.gamma = 5/3; D.sigma = 3.848e-19; D.mu = 4.809e-26;
    return;
end

if strcmp(name,'warm_air')
    D.gamma = 7/5; D.sigma = 3.8483-19; D.mu = 4.809e-26;
    return;
end

if strcmp(name,'10um_iron_balls');
    D.gamma = 1.01; D.sigma = pi*25e-12; D.mu = 3e-11;
    return;
end

% Default model = cold molecular hydrogen
D = fluidDetailModel('cold_molecular_hydrogen');

end
