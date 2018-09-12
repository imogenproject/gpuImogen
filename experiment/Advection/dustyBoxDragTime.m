function Kdrag = dustyBoxDragTime(fluidDetails, rhoG, rhoD, vGas, vDust, gammaGas, Pgas)
% Inefficiently but very readably computes the relative accelerations of the
% gas and dust fluids for a fully general drag law.

thermoGas = fluidDetails(1);
thermoDust= fluidDetails(2);

T = (thermoGas.mass * Pgas) / (rhoG * thermoGas.kBolt);
dv = norm(vGas - vDust, 2);

visc = thermoGas.dynViscosity * (T/298.15)^thermoGas.viscTindex;
Rey = rhoG * dv * sqrt(thermoDust.sigma / pi) / visc;

mfp = thermoGas.mass * (T/298.15)^thermoGas.sigmaTindex / (rhoG * thermoGas.sigma * sqrt(2));
Kn = 2*mfp / sqrt(thermoDust.sigma / pi);

% Use the full calculation for meaningful Re but ignore for extremely low speeds
if Rey > 1e-6
    Fone = computeCdrag(Rey, Kn);
    a = -Fone * .5 * dv^2 * (thermoDust.sigma/4) * (rhoG + rhoD) / thermoDust.mass;
    
    Kdrag = a / dv;
else
    cu = (1 + 1*Kn*(1.142+.558*exp(-.999/Kn)));
    
    Kgrain = -3*visc * sqrt(pi*thermoDust.sigma); % for f = k*dv
    %Kvol = Kgrain * rhoD / thermoDust.mass;
    Kdrag = Kgrain / (thermoDust.mass * cu); % force / (mass * velocity) = drag time constant

end

end

function c = computeCdrag(Rey, Kn)
            c = (24/Rey + 4*Rey^(-1/3) + .44*Rey/(12000+Rey)) / (1 + 1*Kn*(1.142+.558*exp(-.999/Kn)));
end
