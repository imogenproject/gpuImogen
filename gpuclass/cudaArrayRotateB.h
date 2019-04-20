#ifndef CUDA_ARRAY_ROTATE_B_H

int flipArrayIndices(MGArray *phi, MGArray **newArrays, int nArrays, int exchangeCode, cudaStream_t *streamPtrs = NULL, MGArray *tempStorage = NULL);
int matchArrayOrientation(MGArray *of, MGArray *in, MGArray *out, MGArray *tempStorage);

#define CUDA_ARRAY_ROTATE_B_H
#endif
