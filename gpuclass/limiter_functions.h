/*
 * limiter_functions.h
 *
 *  Created on: Apr 7, 2020
 *      Author: erik-k
 */

#ifndef LIMITER_FUNCTIONS_H_
#define LIMITER_FUNCTIONS_H_

__device__ __inline__ double fluxLimiter_VanLeer(double derivL, double derivR)
{
double r;

r = 2.0 * derivL * derivR;
if(r > 0.0) { return r /(derivL+derivR); }

return 0;
}

__device__ __inline__ double fluxLimiter_minmod(double derivL, double derivR)
{
if(derivL * derivR < 0) return 0.0;

if(fabs(derivL) > fabs(derivR)) { return derivR; } else { return derivL; }
}

__device__ __inline__ double fluxLimiter_superbee(double derivL, double derivR)
{
if(derivL * derivR < 0) return 0.0;

if(derivR < derivL) return fluxLimiter_minmod(derivL, 2*derivR);
return fluxLimiter_minmod(2*derivL, derivR);
}

__device__ __inline__ double fluxLimiter_Ospre(double A, double B)
{
double r = A*B;
if(r <= 0.0) return 0.0;

return 1.5*r*(A+B)/(A*A+r+B*B);
}

__device__ __inline__ double fluxLimiter_Zero(double A, double B) { return 0.0; }

/* These differ in that they return _HALF_ of the (limited) difference,
 * i.e. the projection from i to i+1/2 assuming uniform widths of cells i and i+1
 */

// 0.5 * van Leer slope limiter fcn = AB/(A+B)
__device__ __inline__ double slopeLimiter_VanLeer(double derivL, double derivR)
{
double r;

r = derivL * derivR;
if(r > 0.0) { return r /(derivL+derivR); }

return 0;
}

#ifdef FLOATFLUX
__device__ __inline__ float slopeLimiter_minmod(float derivL, float derivR)
{
if(derivL * derivR < 0) return 0.0;

if(fabsf(derivL) > fabsf(derivR)) { return .5*derivR; } else { return .5*derivL; }
}
#else
// .5 * minmod slope limiter fcn = min(A/2,B/2)
__device__ __inline__ float slopeLimiter_minmod(double derivL, double derivR)
{
if(derivL * derivR < 0) return 0.0;

if(fabs(derivL) > fabs(derivR)) { return .5*derivR; } else { return .5*derivL; }
}
#endif

// .5 * superbee slope limiter fcn = ...
__device__ __inline__ double slopeLimiter_superbee(double derivL, double derivR)
{
if(derivL * derivR < 0) return 0.0;

if(derivR < derivL) return fluxLimiter_minmod(derivL, 2*derivR);
return .5*fluxLimiter_minmod(2*derivL, derivR);
}

// 0.5 * ospre slope limiter fcn = .75*A*B*(A+B)/(A^2+AB+B^2)
__device__ __inline__ double slopeLimiter_Ospre(double A, double B)
{
double R = A*B;
if(R > 0) {
	double S = A+B;
	return .75*R*S/(S*S-R);
	}
return 0.0;
}

// 0.5 * van albada limiter fcn = .5*A*B*(A+B)/(A^2+B^2)
__device__ __inline__ double slopeLimiter_vanAlbada(double A, double B)
{
double R = A*B;
if(R < 0) return 0;
return .5*R*(A+B)/(A*A+B*B);
}

// 0.5 * monotized central limiter fcn = min(A, B, (A+B)/4)
__device__ __inline__ double slopeLimiter_MC(double A, double B)
{
	if(A*B <= 0) return 0;
	double R = B/A;
    double S = .25+.25*R;
	//max(0, min(b, .25(a+b), a))

	S = (S < R) ? S : R;
	S = (S < 1) ? S : 1.0;
	return S*A;
}

__device__ __inline__ double slopeLimiter_Zero(double A, double B) { return 0.0; }

#endif /* LIMITER_FUNCTIONS_H_ */
