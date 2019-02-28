
classdef MPI_Syms

properties(Constant = true)

    ML_MPIDOUBLE = 1;
    ML_MPISINGLE = 2;
    ML_MPIUINT16 = 3;
    ML_MPIINT16  = 4;
    ML_MPIUINT32 = 5;
    ML_MPIINT32  = 6;
    ML_MPIUINT64 = 7;
    ML_MPIINT64  = 8;
    ML_MPICHAR   = 9;

    namestrings = {'double', 'single', 'uint16', 'int16', 'uint32', 'int32', 'uint64', 'int64', 'char'};
end

methods (Static = true)

    function s = typename(x)
        if (x >= 1) && (x <= 9) 
            s = MPI_Syms.namestrings{x};
	else
            s = 'INVALID';
	end
    end


end


end
