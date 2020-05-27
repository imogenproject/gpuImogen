// If this is defined, the code will use the explicit trapezoid method
// This choice of quadrature points should yield improved stability properties
#define USE_SSPRK

// If this is defined, the code will use 3rd-order TVD Runge-Kutta
// Implemented and tested, but does not yield any improvement in error metrics
// (surprise, the spatial code is only 2nd order accurate)
//#define USE_RK3

// If neither is defined the code uses explicit midpoint for the fluid step.

#ifdef USE_SSPRK
#define FLUID_METHOD_HALO_SIZE 4;
#else
#ifdef USE_RK3
#define FLUID_METHOD_HALO_SIZE 6;
#else
#define FLUID_METHOD_HALO_SIZE 3;
#endif
#endif
