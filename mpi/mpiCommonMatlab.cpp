
MPI_Datatype typeid_ml2mpi(mxClassID id)
{

switch(id) {
  case mxUNKNOWN_CLASS: return MPI_BYTE; break; /* We're going down anyway */
  case mxCELL_CLASS: return MPI_BYTE; break; /* we're boned */
  case mxSTRUCT_CLASS: return MPI_BYTE; break; /* we're boned */
  case mxLOGICAL_CLASS: return MPI_BYTE; break;
  case mxCHAR_CLASS: return MPI_CHAR; break;
  case mxVOID_CLASS: return MPI_BYTE; break;
  case mxDOUBLE_CLASS: return MPI_DOUBLE; break;
  case mxSINGLE_CLASS: return MPI_FLOAT; break;
  case mxINT8_CLASS: return MPI_BYTE; break;
  case mxUINT8_CLASS: return MPI_BYTE; break;
  case mxINT16_CLASS: return MPI_SHORT; break;
  case mxUINT16_CLASS: return MPI_SHORT; break;
  case mxINT32_CLASS: return MPI_INT; break;
  case mxUINT32_CLASS: return MPI_INT; break;
  case mxINT64_CLASS: return MPI_LONG; break;
  case mxUINT64_CLASS: return MPI_LONG; break;
  case mxFUNCTION_CLASS: return MPI_BYTE; break; /* we're boned */
  }

return MPI_BYTE;
}

mxArray *cIntegerVectorToMatlab(int *v, int numel)
{
	mwSize m = 1;
	mwSize n = numel;
	mxArray *a = mxCreateDoubleMatrix(m, n, mxREAL);

	double *z = mxGetPr(a);
	int j;
	for(j = 0; j < numel; j++) { z[j] = (double)v[j]; }

	return a;
}

mxArray *cDoubleVectorToMatlab(double *v, int numel)
{
	mwSize m = 1;
	mwSize n = numel;
	mxArray *a = mxCreateDoubleMatrix(m, n, mxREAL);

	double *z = mxGetPr(a);
	int j;
	for(j = 0; j < numel; j++) { z[j] = v[j]; }

	return a;
}

/* Converts the input ParallelTopology structure to a Matlab
 * mxArray structure */
int topoCToStructure(ParallelTopology *topo, mxArray **plhs)
{
	mxArray *mlTopology;
	const char  *  topoFields[7]
	                  = {"ndim", "comm", "coord", "neighbor_left", "neighbor_right", "nproc", "dimcomm"};

	mlTopology = plhs[0] = mxCreateStructMatrix(1, 1, 7, topoFields);

	mxSetFieldByNumber(mlTopology, 0, 0, mxCreateDoubleScalar((double)topo->ndim));
	mxSetFieldByNumber(mlTopology, 0, 1, mxCreateDoubleScalar((double)topo->comm));
	mxSetFieldByNumber(mlTopology, 0, 2, cIntegerVectorToMatlab(topo->coord, 3));
	mxSetFieldByNumber(mlTopology, 0, 3, cIntegerVectorToMatlab(topo->neighbor_left, 3));
	mxSetFieldByNumber(mlTopology, 0, 4, cIntegerVectorToMatlab(topo->neighbor_right, 3));
	mxSetFieldByNumber(mlTopology, 0, 5, cIntegerVectorToMatlab(topo->nproc, 3));
	mxSetFieldByNumber(mlTopology, 0, 6, cIntegerVectorToMatlab(topo->dimcomm, 3));

    return 0; // FIXME: check for errors in all this mex API stuff...
}


int topoStructureToC(const mxArray *prhs, ParallelTopology* pt)
{
	mxArray *a;

	a = mxGetField(prhs,0,(const char *)"ndim"); /* n dimensions */
	pt->ndim = (int)*mxGetPr(a);
	a = mxGetField(prhs,0,(const char *)"comm"); /* MPI_Comm_c2f of world comm */
	pt->comm = (int)*mxGetPr(a); // Prevent invalid comm below
	if(pt->comm < 0) pt->comm = MPI_Comm_c2f(MPI_COMM_WORLD);

	double *val;
	int i;

	val = mxGetPr(mxGetField(prhs,0,(const char *)"coord")); /* My coordinate */
	for(i = 0; i < pt->ndim; i++) pt->coord[i] = (int)val[i];

	val = mxGetPr(mxGetField(prhs,0,(const char *)"neighbor_left")); /* left neighbor per dimension */
	for(i = 0; i < pt->ndim; i++) pt->neighbor_left[i] = (int)val[i];

	val = mxGetPr(mxGetField(prhs,0,(const char *)"neighbor_right")); /* right neighbor per dim */
	for(i = 0; i < pt->ndim; i++) pt->neighbor_right[i] = (int)val[i];

	val = mxGetPr(mxGetField(prhs,0,(const char *)"nproc")); /* Number of nodes/dimension */
	for(i = 0; i < pt->ndim; i++) pt->nproc[i] = (int)val[i];

	val = mxGetPr(mxGetField(prhs,0,(const char *)"dimcomm")); /* MPI_Comm_c2f of directional communicators */
	for(i = 0; i < pt->ndim; i++) pt->dimcomm[i] = (int)val[i];

	for(i = pt->ndim; i < TOPOLOGY_MAXDIMS; i++) {
		pt->coord[i] = 0;
		pt->nproc[i] = 1;
	}

	return 0; // FIXME: Check for errors in all this stuff...
}
