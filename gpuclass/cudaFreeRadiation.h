/*
 * cudaFreeRadiation .h
 *
 *  Created on: Jan 11, 2016
 *      Author: erik
 */

#ifndef CUDAFREERADIATION__H_
#define CUDAFREERADIATION__H_

typedef struct __ParametricRadiation {
	double exponent;
	double prefactor;
	double minTemperature;
} ParametricRadiation;

int sourcefunction_OpticallyThinPowerLawRadiation(MGArray *fluid, MGArray *radRate, int isHydro, double gamma, ParametricRadiation *rad);

#endif /* CUDAFREERADIATION__H_ */
