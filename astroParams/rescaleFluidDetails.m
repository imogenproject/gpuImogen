function D = rescaleFluidDetails(F, mass, length, time)

D = F;

%D = struct('gamma',1,'sigma',1,'sigmaTindex',0, 'mass',1.66e-27,'dynViscosity',10e-6,'viscTindex',0.5,'minMass',1,'kBolt',1.381e-23);

D.sigma = F.sigma / length^2;
D.mass =  F.mass / mass;
D.dynViscosity = F.dynViscosity * length * time / mass;
D.minMass = F.minMass / mass;
D.kBolt        = F.kBolt * time^2 / (mass * length^2);
if D.Cisothermal ~= -1
    D.Cisothermal = D.Cisothermal * time / length;
end

end