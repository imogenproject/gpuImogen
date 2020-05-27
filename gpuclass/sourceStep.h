#ifndef SOURCESTEPH_
#define SOURCESTEPH_

int performSourceFunctions(int srcType, GridFluid *fluids, int nFluids, FluidStepParams fsp, ParallelTopology *topo, GravityData *gravdata, ParametricRadiation *prad, MGArray *tmpStorage);

#endif
