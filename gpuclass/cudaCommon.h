typedef struct {
        double *fluidIn[5];
        double *fluidOut[5];

        double *B[3];

        double *Ptotal;
        double *cFreeze;
        } fluidVarPtrs;


typedef struct {
    int ndims;
    int dim[3];
    int numel;
    } ArrayMetadata;

double **getGPUSourcePointers(const mxArray *prhs[], ArrayMetadata *metaReturn, int fromarg, int toarg);
double **makeGPUDestinationArrays(int64_t *reference, mxArray *retArray[], int howmany);

