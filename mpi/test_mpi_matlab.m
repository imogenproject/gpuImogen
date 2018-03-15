function test_mpi(rez)

if nargin < 1;
    rez = [32 32 32];
    disp('Defaulting to resolution of 32^3');
end

context = parallel_start();
topology = parallel_topology(context, 3);
pg = ParallelGlobals(context, topology); % yucky global from hell

MYID = context.rank;

geometry = GeometryManager(rez);

x = ones([5 1])*MYID;
xout = mpi_allgather(x);

if MYID == 0; fprintf('FIXME: didnt actually test mpi_allgather...\n'); end

mpi_barrier();
if MYID == 0; fprintf('TESTING REDUCTION FUNCTIONS ------------\n'); end
rng(MYID); res = 400;
alpha = rand(res);
beta  = single(alpha);
gamma = int32(round(alpha));

% TEST PARALLEL all() (aka MPI_BAND)
A = mpi_all(round(alpha));
B = mpi_all(round(beta));
C = mpi_all(gamma);

if MYID == 0
    rng(0);
    trueans = rand(res);
    trueans = round(trueans);
    for n = 2:context.size;
        rng(n-1); tst = rand(res);
        trueans = trueans .* round(tst);
    end
    trueans = (trueans ~= 0);
    fail = any(trueans(:) - A(:)) | any(trueans(:) - B(:)) | any(int32(trueans(:)) - C(:));
    if fail; fprintf('Tested MPI_LAND. Result: FAILURE!\n'); else fprintf('Tested MPI_LAND; Result: Success.\n'); end
end

% TEST PARALLEL any() (aka MPI_BOR)
A = mpi_any(round(alpha));
B = mpi_any(round(beta));
C = mpi_any(gamma);

if MYID == 0;
    rng(0);
    trueans = rand(res);
    trueans = round(trueans);
    for n = 2:context.size;
        rng(n-1); tst = rand(res);
        trueans = trueans + round(tst);
    end
    trueans = (trueans ~= 0);
    fail = any(trueans(:) - A(:)) | any(trueans(:) - B(:)) | any(int32(trueans(:)) - C(:));
    if fail; fprintf('Tested MPI_LOR. Result: FAILURE!\n'); else fprintf('Tested MPI_LOR; Result: Success.\n'); end
end

% TEST MAX(x)
A = mpi_max(alpha);
B = mpi_max(beta);
C = mpi_max(gamma);

if MYID == 0
    rng(0);
    trueans = rand(res);
    for n = 2:context.size
        rng(n-1); 
        trueans = max(trueans, rand(res));
    end

    fail = any(trueans(:) - A(:)) | any(abs(trueans(:) - B(:)) > 1e-6) | any(int32(trueans(:)) - C(:));
    if fail; fprintf('Tested MPI_MAX. Result: FAILURE!\n'); else fprintf('Tested MPI_MAX; Result: Success.\n'); end
end

% TEST MIN(x)
A = mpi_min(alpha);
B = mpi_min(beta);
C = mpi_min(gamma);

if MYID == 0
    rng(0);
    trueans = rand(res);
    for n = 2:context.size
        rng(n-1);
        trueans = min(trueans, rand(res));
    end
    fail = any(trueans(:) - A(:)) | any(abs(trueans(:) - B(:)) > 1e-6) | any(int32(trueans(:)) - C(:));
    if fail; fprintf('Tested MPI_MIN. Result: FAILURE!\n'); else fprintf('Tested MPI_MIN; Result: Success.\n'); end
end

% TEST MPI_PROD
A = mpi_prod(alpha);
B = mpi_prod(beta);
C = mpi_prod(gamma);

if MYID == 0
    rng(0);
    trueans = rand(res);
    for n = 2:context.size
        rng(n-1);
        trueans = trueans .* rand(res);
    end
    fail = any(trueans(:) - A(:)) | any(abs(trueans(:) - B(:)) > 1e-6);
    if fail; fprintf('Tested MPI_PROD. Result: FAILURE!\n'); else fprintf('Tested MPI_PROD; Result: Success.\n'); end
end

% TEST MPI_SUM
A = mpi_sum(alpha);
B = mpi_sum(beta);
C = mpi_sum(gamma);

if MYID == 0
    rng(0);
    trueans = rand(res);
    for n = 2:context.size
        rng(n-1);
        trueans = trueans + rand(res);
    end
    fail = any(trueans(:) - A(:)) | any(abs(trueans(:) - B(:)) > 1e-6);
    if fail; fprintf('Tested MPI_SUM. Result: FAILURE!\n'); else fprintf('Tested MPI_SUM; Result: Success.\n'); end
end

mpi_barrier();

mpi_deleteDimcomm(geometry.topology);

%mpi_finalize();

end
