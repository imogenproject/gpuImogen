#include "stdio.h"
#include "stdlib.h"

#include "mpi.h"
#include "math.h"

#include "mex.h"

#include "mpi_common.h"

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

/*mxClassID typeid_mpi2ml(MPI_Datatype md)
{
switch((int)md) {
  case MPI_BYTE: return mxCHAR_CLASS; break;
  case MPI_PACKED: return mxCHAR_CLASS; break;
  case MPI_CHAR: return mxCHAR_CLASS; break;
  case MPI_SHORT: return mxINT16_CLASS; break;
  case MPI_INT: return mxINT32_CLASS; break;
  case MPI_LONG: return mxINT64_ClASS; break;
  case MPI_FLOAT: return mxFLOAT_CLASS; break;
  case MPI_DOUBLE: return mxDOUBLE_CLASS; break;
  case MPI_LONG_DOUBLE: return MPI_BYTE; break;  we're scrwed 
  case MPI_UNSIGNED_CHAR: return mxCHAR_CLASS; break;
  }
}*/
