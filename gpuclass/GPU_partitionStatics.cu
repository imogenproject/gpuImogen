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
void reassembleStaticsLists(singleStaticList *nulists, singleStaticList *forpart, double *nuoffsets);
int3 permutateIndices(int3 *in, int dir);

int vomitDebug;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if((nlhs != 1) || (nrhs != 3)) { mexErrMsgTxt("Form: new_offsets_vector = GPU_partitionStatics(fluid array, statics array, offsetvector)\n"); }

	CHECK_CUDA_ERROR("entering GPU_upload");

	MGArray main, stats;
	double *offsetVector; // [OS0 N0 OS1 N1 OS2 N2 ... OS5 N5]'

	// If this is turned on, prepare for the console to be hosed down by
	// debug info...
	vomitDebug = 1;

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
		if(main.nGPUs == 1) return; // no partitioning problem can exist
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

	if(vomitDebug) { printf("Filtering static lists...\n"); }
	for(dirs = 0; dirs < 6; dirs++) {
		for(parts = 0; parts < main.nGPUs; parts++) {
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

		reassembleStaticsLists(&newStatics[6*parts], &finalnewlist, newost + 12*parts);
		for(dirs = 0; dirs < 6; dirs++) { free(newStatics[6*parts+dirs].list); }

		if(finalnewlist.nstatics > 0) {
			if(vomitDebug) { printf("Assembled statics for partition %i have %i elements: Reuploading to GPU\n", parts, finalnewlist.nstatics); }

			cudaSetDevice(main.deviceID[parts]);
			cudaMemcpy((void *)stats.devicePtr[parts], finalnewlist.list, 3* finalnewlist.nstatics * sizeof(double), cudaMemcpyHostToDevice);
			free(finalnewlist.list);
		}

	}

	int nq;
	if(vomitDebug) { printf("New offset vectors:"); }
	for(nq = 0; nq < 12*main.nGPUs; nq++) {
		if(nq % 12 == 0) printf("\n");
		if(vomitDebug) printf("%i ", (int)newost[nq]);
	}
	if(vomitDebug) printf("\n");


	return;
}

/* Given the main array *bigarray, the original wad of statics (copied back to cpu...) described by *origs,
 * a list of 6*(# partitions) sublists to write to, and the original offset/size vector *offsets,
 * makes 6*nPartitions copies into sublists
 *
 * Each has its own malloc()ed array containing only the original statics for that direction. They are
 * emitted from here the same for each partition and need to be filtered for addresses in the range of each
 * partition to be valid.
 */
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

/* Given the main array *bigarray and a single lists of statics for it *full,
 * runs through all full->nstatics elements in full->list (size(full->list) = [nstatics 3]),
 * copies only those which on the gpu live on partition full->part, overwriting the
 * original contents and metainformation in full.
 */
void filterStaticList(MGArray *bigarray, singleStaticList *full)
{
if(full->list == NULL) return; // tarry thee not...

int i, j, n;

n = full->nstatics;
int sub[6];

// get partition data
calcPartitionExtent(bigarray, full->part, &sub[0]);

fadeElement f;

int3 dima = makeInt3(bigarray->dim[0], bigarray->dim[1], bigarray->dim[2]);
int3 dimfull = permutateIndices(&dima, full->direct);
int3 ra;
j = 0;
// count how many elements will remain...
for(i = 0; i < n; i++) {
	f.linearIdx = full->list[i];
	cvtLinearToTriple(&f, &dimfull);
	ra = permutateIndices(&f.idx, full->direct);

	if((ra.x >= sub[0]) && (ra.x < (sub[0]+sub[3]))
			&& (ra.y >= sub[1]) && (ra.y < (sub[1]+sub[4]))
			&& (ra.z >= sub[2]) && (ra.z < (sub[2]+sub[5]))
			) j++;
}

double *q = (double *)malloc(full->nstatics * 3 * sizeof(double));

int nnew = j;
j = 0;

// The dimensions of the full array, permutated in appropriate order
int3 dimperm = dimfull;

// the dimensions of the partition, permutated in appropriate order for rebuilding linear indexes
int3 subunperm = makeInt3(sub[3], sub[4], sub[5]);
int3 subperm = permutateIndices(&subunperm, full->direct);

for(i = 0; i < n; i++) {
	f.linearIdx = full->list[i];
	cvtLinearToTriple(&f, &dimperm);

	// flip back to normal order
	ra = permutateIndices(&f.idx, full->direct);

	if((ra.x >= sub[0]) && (ra.x < (sub[0]+sub[3]))
				&& (ra.y >= sub[1]) && (ra.y < (sub[1]+sub[4]))
				&& (ra.z >= sub[2]) && (ra.z < (sub[2]+sub[5]))
	) {
		//ra = permutateIndices(&f.idx, full->direct);
		ra = f.idx;

		q[j]        = ra.x + subperm.x*(ra.y + subperm.y*ra.z);
		q[nnew+j]   = full->list[n+i];
		q[2*nnew+j] = full->list[2*n+i];

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
void reassembleStaticsLists(singleStaticList *nulists, singleStaticList *forpart, double *nuoffsets)
{
int dirct;
int ntotal = 0;

for(dirct = 0; dirct < 6; dirct++) {
	ntotal += nulists[dirct].nstatics;
}

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

