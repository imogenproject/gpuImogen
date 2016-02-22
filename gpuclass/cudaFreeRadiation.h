/*
 * cudaFreeRadiation .h
 *
 *  Created on: Jan 11, 2016
 *      Author: erik
 */

#ifndef CUDAFREERADIATION__H_
#define CUDAFREERADIATION__H_

int sourcefunction_OpticallyThinPowerLawRadiation(MGArray *fluid, MGArray *radRate, int isHydro, double gamma, double exponent, double prefactor, double minimumTemperature);

#endif /* CUDAFREERADIATION__H_ */
