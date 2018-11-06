#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#ifdef UNIX
#include <stdint.h>
#include <unistd.h>
#endif
#include "mex.h"

// CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas.h"

#include "cudaCommon.h"

typedef struct fadeElement {
	long linearIdx;
	int3 idx;
	double cst, val;
} __fadeElement;

typedef struct singleStaticList {
	double *list;
	int nstatics; // list is [nstatics][3] in size: [addr c f], each row

	int direct, part;
} __singleStaticList;

void cvtLinearToTriple(fadeElement *f, int3 *dim);
void disassembleStaticsLists(MGArray *bigarray, singleStaticList *origs, singleStaticList *sublists, double *offsets);
void filterStaticList(MGArray *bigarray, singleStaticList *full);
void reassembleStaticsLists(singleStaticList *nulists, singleStaticList *forpart, double *nuoffsets, int originalStaticNx);
int3 permutateIndices(int3 *in, int dir);

int vomitDebug;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if((nlhs != 1) || (nrhs != 3)) { mexErrMsgTxt("Form: new_offsets_vector = GPU_partitionStatics(fluid array, statics array, offsetvector)\n"); }

	CHECK_CUDA_ERROR("entering GPU_upload");

	MGArray main, stats;
	double *offsetVector; // [OS0 N0 OS1 N1 OS2 N2 ... OS5 N5]'

	// If this is turned on, prepare for the console to be hosed down by
	// debug info...
	vomitDebug = 0;

	int status = MGA_accessMatlabArrays(prhs, 0, 0, &main);
	if(status != SUCCESSFUL) { DROP_MEX_ERROR("Unable to access fluid array!\n"); }

	status = MGA_accessMatlabArrays(prhs, 1, 1, &stats);
	if(status != SUCCESSFUL) { DROP_MEX_ERROR("Unable to access statics array!\n"); }

	if(vomitDebug) {
		printf("BEGIN MASSIVE, VOMITOUS DUMP OF DEBUGGING INFORMATION FOR GPU_partitionStatics.cu:\n");
		printf("About main input array\n");
		MGA_debugPrintAboutArray(&main);
		printf("About input statics array\n");
		MGA_debugPrintAboutArray(&stats);
	}

	offsetVector = mxGetPr(prhs[2]);
	if(vomitDebug) {
		printf("original offset vector: ");
		int dj;
		for(dj = 0; dj < 12; dj++) { printf("%i ", (int)offsetVector[dj]); }
		printf("\n");
	}

	if(vomitDebug == 0) {
		if(stats.nGPUs == 0) { // no statics in use
                        // zero gpus = no statics in use on this node
                        mwSize dims[2];
                        dims[0] = 12*main.nGPUs;
                        dims[1] = 1;
                        plhs[0] = mxCreateNumericArray(2, &dims[0], mxDOUBLE_CLASS, mxREAL);

                        double *newost = mxGetPr(plhs[0]);
                        int i;
                        for(i = 0; i < dims[0]; i++) { newost[i] = 0; }

                        return; // no partitioning problem can exist

		}
		if(main.nGPUs <= 1) {
			// one gpu = only one partition so this is a passthru
			// zero gpus = no statics in use on this node
			mwSize dims[2];
	        	dims[0] = 12;
        		dims[1] = 1;
		        plhs[0] = mxCreateNumericArray(2, &dims[0], mxDOUBLE_CLASS, mxREAL);

	        	double *newost = mxGetPr(plhs[0]);
			int i;
			for(i = 0; i < 12; i++) { newost[i] = offsetVector[i]; }

			return; // no partitioning problem can exist
		}
	} else {
		if(main.nGPUs == 1) {
		printf("DEBUG NOTICE: Only one GPU in use, but partitioner will run anyway\n");
		printf("DEBUG NOTICE: THIS MUST YIELD AN IDENTITY OPERATION!!!!!\n");
		}
	}

	singleStaticList origStats;
	origStats.list = NULL;

	MGA_downloadArrayToCPU(&stats, &origStats.list, 0);

	singleStaticList newStatics[6*main.nGPUs];

	if(vomitDebug) { printf("Entering dissassembleStaticsLists...\n"); }
	disassembleStaticsLists(&main, &origStats, &newStatics[0], offsetVector);

	int parts, dirs;

	for(dirs = 0; dirs < 6; dirs++) {
		for(parts = 0; parts < main.nGPUs; parts++) {
			if(vomitDebug) { printf("Filtering static lists for dir=%i part=%i...\n", dirs, parts); }
			filterStaticList(&main, &newStatics[dirs + 6*parts]);
		}
	}

	mwSize dims[2];
	dims[0] = 12;
	dims[1] = main.nGPUs;
	plhs[0] = mxCreateNumericArray(2, &dims[0], mxDOUBLE_CLASS, mxREAL);

	double *newost = mxGetPr(plhs[0]);

	singleStaticList finalnewlist;

	for(parts = 0; parts < main.nGPUs; parts++) {
		finalnewlist.part = parts;

		if(vomitDebug) { printf("Static list reassembly in progress for partition %i...\n", parts); }

		reassembleStaticsLists(&newStatics[6*parts], &finalnewlist, newost + 12*parts, stats.dim[0]);
		for(dirs = 0; dirs < 6; dirs++) { free(newStatics[6*parts+dirs].list); }

		if(finalnewlist.nstatics > 0) {
			if(vomitDebug) { printf("Assembled statics for partition %i have %i elements: Reuploading to GPU\n", parts, finalnewlist.nstatics); }

			cudaSetDevice(stats.deviceID[parts]);
			CHECK_CUDA_ERROR("set device");
			cudaMemcpy((void *)stats.devicePtr[parts], finalnewlist.list, 3* finalnewlist.nstatics * sizeof(double), cudaMemcpyHostToDevice);
			CHECK_CUDA_ERROR("cudaMemcpy");
			cudaDeviceSynchronize();
			CHECK_CUDA_ERROR("cudaDeviceSynchronize");
			free(finalnewlist.list);
		}

	}

	int nq;
	if(vomitDebug) {
		printf("New offset vectors:"); 
		for(nq = 0; nq < 12*main.nGPUs; nq++) {
			if(nq % 12 == 0) printf("\n");
			printf("%i ", (int)newost[nq]);
		}
		printf("\n");
	}


	return;
}

/* Given the main array *bigarray, the original partition-unaware statics (copied back to cpu...) described by *origs,
 * a list of 6*(# partitions) sublists to write to, and the original offset/size vector *offsets, makes 6*nPartitions
 * copies into sublists[i]
 *
 * Each has its own malloc()ed array containing only the original statics for that direction. They are
 * emitted from here the same for each partition; No range check or address recalculation is done. */
void disassembleStaticsLists(MGArray *bigarray, singleStaticList *origs, singleStaticList *sublists, double *offsets)
{
int dir, part, n, i;

int nxOrig = (int)(offsets[10]+offsets[11]);

double *s;

// for each of 6 direction permutations,
for(dir = 0; dir < 6; dir++) {
	// for each partition,
	for(part = 0; part < bigarray->nGPUs; part++) {

		// allocate storage for the full set of statics (the most possiblly needed)
		n = offsets[2*dir+1];

		// if there are any,
		if(n > 0) {
			sublists[dir+6*part].list = s = (double *)malloc(offsets[2*dir+1]*3*sizeof(double));

			// and copy original(offset:(offset+n),:) to sub(:,:)
			for(i = 0; i < n; i++) {
				s[i]     = origs->list[(int)(offsets[2*dir] + i)];
				s[n+i]   = origs->list[(int)(offsets[2*dir] + i + nxOrig)];
				s[2*n+i] = origs->list[(int)(offsets[2*dir] + i + 2*nxOrig)];
			}
		} else {
			// or assign NULL and skip
			sublists[dir+6*part].list = NULL;
		}

		// copy other relevant metainfo
		sublists[dir+6*part].direct = dir;
		sublists[dir+6*part].nstatics = n;
		sublists[dir+6*part].part = part;
	}
}

}

int tupleIsInPartition(int3 *tuple, int *sub, int direct)
{

int3 q = makeInt3(sub[0], sub[1], sub[2]);
int3 os = permutateIndices(&q, direct);

     q = makeInt3(sub[3], sub[4], sub[5]);
int3 s = permutateIndices(&q, direct);

if( (tuple->x >= os.x) && (tuple->x < (os.x+s.x)) && (tuple->y >= os.y) && (tuple->y < (os.y+s.y)) && (tuple->z >= os.z) && (tuple->z < (os.z + s.z))) return 1;

return 0;
}

/* Given the main array *bigarray and a single lists of statics for it in *full, runs through all full->nstatics
 * elements in full->list (size(full->list) = [nstatics 3]). Only elements whose tuple lies within the subset
 * identified by calcPartitionExtent(bigarray, full->part) are kept, and their addresses recomputed.
 */
void filterStaticList(MGArray *bigarray, singleStaticList *full)
{
if(full->list == NULL) return; // tarry thee not...

int i, j, n;

n = full->nstatics;
int sub[6];

// get partition size: sub = [zero_x, zero_y, zero_z, n_x, n_y, n_z]
calcPartitionExtent(bigarray, full->part, &sub[0]);

fadeElement f;

// Fetch full dimensions that were originally used to compute linear addresses
// This is always in [1 2 3] permutation.
int3 dima = makeInt3(bigarray->dim[0], bigarray->dim[1], bigarray->dim[2]);

// rotate to reflect the direction for these statics (e.g. 3 1 2 if direct=4)
int3 dimfull = permutateIndices(&dima, full->direct);
int3 ra;
j = 0;
// count how many elements will remain...
for(i = 0; i < n; i++) {
	// fetch the original linear index & convert back to tuple using whole array dims
	// f.idx is now in permuted order
	// suppose cell [5 0 0]_123 is static: if direct=4 then idx=5*nz;
	f.linearIdx = full->list[i]; // suppose this is 5*nz
	cvtLinearToTriple(&f, &dimfull); // this returns [0 5 0]

	// This permutes the subset indices the same as f is permuted and checks range
	if(tupleIsInPartition(&f.idx, &sub[0], full->direct)) j++;

// bad code
//	ra = permutateIndices(&f.idx, full->direct); // ra = [0 0 5]
//
//	if((ra.x >= sub[0]) && (ra.x < (sub[0]+sub[3]))
//			&& (ra.y >= sub[1]) && (ra.y < (sub[1]+sub[4]))
//			&& (ra.z >= sub[2]) && (ra.z < (sub[2]+sub[5]))
//			) j++;
}

double *q = (double *)malloc(full->nstatics * 3 * sizeof(double));

int nnew = j;
j = 0;

// The dimensions of the full array, permutated in appropriate order
int3 dimperm = dimfull;

// the dimensions of the partition, permutated in appropriate order for rebuilding linear indexes
int3 subunperm = makeInt3(sub[3], sub[4], sub[5]);
int3 subperm = permutateIndices(&subunperm, full->direct);

// and of the offset
int3 osunperm = makeInt3(sub[0], sub[1], sub[2]);
int3 osperm = permutateIndices(&osunperm, full->direct);

int imax = sub[3]*sub[4]*sub[5];

if(vomitDebug) { printf("There are %i statics in this set. dir=%i\npermutated offset = [%i %i %i]\npermutated dims=[%i %i %i]\n", nnew, full->direct, osperm.x, osperm.y, osperm.z, subperm.x, subperm.y, subperm.z); }

for(i = 0; i < n; i++) {
	f.linearIdx = full->list[i];
	cvtLinearToTriple(&f, &dimperm);
	// f.idx is in permutated order

        if(tupleIsInPartition(&f.idx, &sub[0], full->direct)) {
		// subtract partition offset
		ra.x = f.idx.x - osperm.x;
		ra.y = f.idx.y - osperm.y;
		ra.z = f.idx.z - osperm.z;

		// compute partition's index
		q[j]        = ra.x + subperm.x*(ra.y + subperm.y*ra.z);

		// and copy the fade value & coefficient
		q[nnew+j]   = full->list[n+i];
		q[2*nnew+j] = full->list[2*n+i];

if((q[j] >= imax) || (q[j] < 0)) {
printf("GPU_partitionStatics: OH CRAP!!! i=%i, j=%i, dir=%i, part=%i, new %i > %i\n", i, j-1, full->direct, full->part, (int)q[j], imax);
printf("linear index = %i, f.idx = [%i %i %i], subperm=[%i %i %i], osperm=[%i %i %i]\n", f.linearIdx, f.idx.x, f.idx.y, f.idx.z, subperm.x, subperm.y, subperm.z, osperm.x, osperm.y, osperm.z);
}
		j++;
	}

}

memcpy(full->list, q, nnew*3*sizeof(double));
full->nstatics = nnew;
free(q);

}

/* Reads nulists[0] to nulists[5] (the 6 directions) and and merges all 6 lists into
 * forpart->list[] and outputs appropriate #s and offsets into nuoffests[0..11]
 */
void reassembleStaticsLists(singleStaticList *nulists, singleStaticList *forpart, double *nuoffsets, int originalStaticNx)
{
int dirct;
int ntotal = 0;

//for(dirct = 0; dirct < 6; dirct++) {
//	ntotal += nulists[dirct].nstatics;
//}
ntotal = originalStaticNx; // Just go with it

forpart->list = (double *)malloc(ntotal*3*sizeof(double));
forpart->nstatics = ntotal;

int offset = 0;
int n, i;

for(dirct = 0; dirct < 6; dirct++) {
	// stuff
	n = nulists[dirct].nstatics;
	for(i = 0; i < n; i++) {
		forpart->list[         offset+i] = nulists[dirct].list[    i];
		forpart->list[ntotal  +offset+i] = nulists[dirct].list[n  +i];
		forpart->list[2*ntotal+offset+i] = nulists[dirct].list[2*n+i];
	}

	nuoffsets[2*dirct]   = offset;
	nuoffsets[2*dirct+1] = n;

	offset += n;
}

}

void cvtLinearToTriple(fadeElement *f, int3 *dim)
{
	long p = f->linearIdx;
	long nxy = dim->x * dim->y;

	long q = p / nxy;

	f->idx.z = q;
	p -= q * nxy;

	q = p / dim->x;

	f->idx.y = q;

	p = p - q*dim->x;

	f->idx.x = p;
}

int3 permutateIndices(int3 *in, int dir)
{
	//permGroup = [1 2 3; 1 3 2; 2 1 3; 2 3 1; 3 1 2; 3 2 1];

int3 out;
switch(dir) {
case 0: out.x = in->x; out.y = in->y; out.z = in->z; break;
case 1: out.x = in->x; out.y = in->z; out.z = in->y; break;
case 2: out.x = in->y; out.y = in->x; out.z = in->z; break;
case 3: out.x = in->y; out.y = in->z; out.z = in->x; break;
case 4: out.x = in->z; out.y = in->x; out.z = in->y; break;
case 5: out.x = in->z; out.y = in->y; out.z = in->x; break;
}

return out;

}

