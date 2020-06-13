/*
 * cudaTestSourceComposite.h
 *
 *  Created on: Apr 1, 2020
 *      Author: erik-k
 */

#ifndef CUDATESTSOURCECOMPOSITE_H_
#define CUDATESTSOURCECOMPOSITE_H_

int sourcefunction_Composite(MGArray *fluid, MGArray *phi, MGArray *XYVectors, GeometryParams geom, double rhoNoG, double rhoFullGravity, double dt, int spaceOrder, int temporalOrder, MGArray *storageBuffer);


#endif /* CUDATESTSOURCECOMPOSITE_H_ */
