#ifndef CUDASOURCE2FLUIDDRAG_H_

int sourcefunction_2FluidDrag(MGArray *fluidA, MGArray *fluidB, GeometryParams *geo, ThermoDetails *thermogas, ThermoDetails *thermodust, double dt, int method);

#else
#define CUDASOURCE2FLUIDDRAG_H_
#endif
