__global__ void cukern_DirectionalMax(double *d1, double *d2, double *out, int direct, int nx, int ny, int nz);
__global__ void cukern_GlobalMax(double *din, int n, double *dout);
__global__ void cukern_GlobalMax_forCFL(double *rho, double *cs, double *px, double *py, double *pz, int n, double *dout, int *dirOut);


