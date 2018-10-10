#ifndef CUDA_ARRAY_ROTATE_B_H

int flipArrayIndices(MGArray *phi, MGArray **newArrays, int nArrays, int exchangeCode, cudaStream_t *streamPtrs = NULL);
//int flipArrayIndices(MGArray *phi, MGArray **newArrays, int nArrays, int exchangeCode);

__global__ void cukern_ArrayTranspose2D(double *src, double *dst, int nx, int ny);
__global__ void cukern_ArrayExchangeXY(double *src,  double *dst, int nx, int ny, int nz);
__global__ void cukern_ArrayExchangeXZ(double *src,  double *dst, int nx, int ny, int nz);
__global__ void cukern_ArrayExchangeYZ(double *src,  double *dst, int nx, int ny, int nz);
__global__ void cukern_ArrayRotateRight(double *src, double *dst, int nx, int ny, int nz);
__global__ void cukern_ArrayRotateLeft(double *src,  double *dst, int nx, int ny, int nz);

#define CUDA_ARRAY_ROTATE_B_H
#endif
