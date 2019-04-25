#!/bin/bash

# --- Test for results directory --- #
resDir="${HOME}/Results"
if [ -d $resDir ]; then
  activeResDir=$resDir
else
  mkdir "${HOME}/Results/"
  echo ">>> WARNING: No Results directory found. One has been created."
fi

if [ -z "${1}" ]; then
  echo ">>> ERROR: No input specified. How to run (./imogen help to see this):"
  ./imogen help
  exit 1
fi

if [ ${1} = "help" ]; then
  echo "       /=============\\"
  echo "+----=< *IMOGEN HELP* >=======----------------------------------------"
  echo "|      \\=============/"
  echo "| Serial operation: ./imogen serial   runfile streamnumber GPUn"
  echo "| Local parallel:   ./imogen parallel runfile streamnumber GPUs #procs host_file"
  echo "| Cluster submit:   ./imogen cluster  runfile streamnumber [GPUs [nodes [ppn [queue [np]]]]]"
  echo "|                   (tees realtime output to ~/imogenout)"
  echo "| Talapas submit:   ./imogen talapas  runfile streamnumber #gpus_per_node #nodes queue [PPN]"
  echo "|"
  echo "|    Serial and local parallel operation execute immediately; Cluster"
  echo "|    operation writes a script and submits via qsub."
  echo "+==============================---------------------------------------"

  exit 0
fi

# --- Check for input file argument --- #
if [ -z "${2}" ]; then
  echo ">>> ERROR: Only one argument; See './imogen help'; Suggest ./imogen serial ${2} 0 0"
  exit 1
fi
RUNFILE=${2}

# --- Determine stream output --- #
stream=${3}
if [ -z "$3" ]; then
  stream="1"
fi
redirectStr="${HOME}/Results/logfile${stream}.out"

if [ -f "$redirectStr" ]; then
  echo ">>> NOTE: requested output stream exists; Appending."
fi


if [ ${1} = "serial" ]; then
  # ./imogen serial runfile stream GPU#
  gpuno=${4}
  if [ -z "$4" ]; then
    gpuno=0
    echo ">>> WARNING: defaulting to GPUs = [0]"
  fi

  echo "parImogenLoad('${RUNFILE}','${redirectStr}','${alias}',$gpuno);" | nohup nice matlab -nodesktop >> $redirectStr & 
  exit 0
fi

if [ ${1} = "parallel" ]; then
  # ./imogen parallel runfile stream# GPUs nprocs hostfile
  gpuno=${4}
  if [ -z "$4" ]; then
    gpuno=0
    echo ">>> WARNING: Parallel run not given GPU sets. Defaulting to $gpuno"
  fi

  numproc=${5}
  if [ -z "$numproc" ]; then
    numproc=1
    echo ">>> WARNING: Parallel run not given #processes (???). Defaulting to np=1"
  fi

  hostFile=${6}
  if [ -z "$hostFile" ]; then
    echo ">>> WARNING: Parallel run not given hostfile. Defaulting to 'localhost'"
    echo "localhost" > fakeHostfile
    hostFile="fakeHostfile"
  fi

  mpirun -np $numproc -bynode matlab -nodisplay -nojvm -r "parImogenLoad('${RUNFILE}','${redirectStr}','${alias}', $gpuno);" >> $redirectStr &

  exit 0
fi

if [ ${1} = "parprofile" ]; then
  # ./imogen parallel runfile stream# GPUs nprocs hostfile
  gpuno=${4}
  if [ -z "$4" ]; then
    gpuno=0
    echo ">>> WARNING: Parallel run not given GPU sets. Defaulting to $gpuno"
  fi

  numproc=${5}
  if [ -z "$numproc" ]; then
    numproc=1
    echo ">>> WARNING: Parallel run not given #processes (???). Defaulting to np=1"
  fi

  hostFile=${6}
  if [ -z "$hostFile" ]; then
    echo ">>> WARNING: Parallel run not given hostfile. Defaulting to 'localhost'"
    echo "localhost" > fakeHostfile
    hostFile="fakeHostfile"
  fi

  mpirun -np $numproc -bynode --hostfile $hostFile nvprof -o gpuimogen.%q{OMPI_COMM_WORLD_RANK}.nvvp matlab -nodisplay -nojvm -r "parImogenLoad('${RUNFILE}','${redirectStr}','${alias}', $gpuno);" >> $redirectStr &

  exit 0
fi


if [ ${1} = "cluster" ]; then
# ./imogen cluster runfile stream #nodes PPN [np] queue
  gpuset=${4};
  if [ -z "$4" ]; then
    gpuset="[0]";
    echo ">>> WARNING: Cluster run not given GPU sets. Defaulting to $gpuset"
  fi

  nnodes=${5};
  if [ -z "$5" ]; then
    nnodes="1";
    echo ">>> WARNING: Cluster run not given #nodes: Defaulting to $nnodes"
  fi

  procpn=${6};
  if [ -z "$6" ]; then
    procpn="1";
    echo ">>> WARNING: Cluster run not given #procs/node. Defaulting to $procpn"
  fi

  QUEUE=${7}
  if [ -z "$7" ]; then
    QUEUE="gpu";
    echo ">>> WARNING: Cluster run not given queue. Defaulting to $QUEUE"
  fi

  NP=${8}
  if [ -z "$8" ]; then
    echo ">>> NOTE: NP not explicitly set, using #nodes x procs/node."
    NP=$(expr $nnodes \* $procpn);
  fi


  TFILE=$(mktemp);

  echo "module load matlab/r2012b" >> $TFILE
  echo "module load mpi-tor/openmpi-1.5.4_gcc-4.5.3" >> $TFILE
  echo "module load cuda/5.5" >> $TFILE
  echo "echo PATH: \$PATH" >> $TFILE
  echo "cd $(pwd)" >> $TFILE
  echo "mpirun --mca btl_tcp_if_include torbr --bynode -np $NP matlab -nodisplay -nojvm -r \"parImogenLoad('${RUNFILE}','${redirectStr}','${alias}', $gpuset);\" " >> $TFILE

fi


if [ ${1} = "talapas" ]; then
# ./imogen talapas runfile stream gpus_per_node #nodes queue [PPN]
  gpureq=${4};
  if [ -z "$4" ]; then
    gpureq="1";
    echo ">>> WARNING: Talapas run not told how many GPUs to request. Requesting 1."
  fi

  nnodes=${5};
  if [ -z "$5" ]; then
    nnodes="1";
    echo ">>> WARNING: Talapas run not given #nodes: Defaulting to $nnodes"
  fi

  QUEUE=${6}
  if [ -z "$6" ]; then
    QUEUE="gpu";
    echo ">>> WARNING: Talapas run not given queue. Defaulting to $QUEUE"
  fi

  procpn=${7};
  if [ -z "$7" ]; then
    procpn="1";
    echo ">>> WARNING: Talapas run defaulting to $procpn processes/node: This is usually correct"
  fi

  NP=$(expr $nnodes \* $procpn);
  echo ">>> Computed $NP total procs to request."

  TFILE=$(mktemp);

  echo "#!/bin/bash" > $TFILE
  echo "#SBATCH --partition=gpu      ### Quality of Service (like a queue in PBS)" >> $TFILE
  echo "#SBATCH --job-name=imogen  ### Job Name" >> $TFILE
  echo "#SBATCH --time=00:10:00      ### Walltime" >> $TFILE
  echo "#SBATCH --nodes=$nnodes             ### Number of Nodes" >> $TFILE
  echo "#SBATCH --ntasks-per-node=$procpn ### Number of tasks (MPI processes)" >> $TFILE
  echo "#SBATCH --gres=gpu:$gpureq      ### General REServation of gpu:number of gpus" >> $TFILE

  echo "module load matlab" >> $TFILE
  echo "module load openmpi" >> $TFILE
  echo "module load cuda" >> $TFILE
  echo "echo PATH: \$PATH" >> $TFILE
  echo "cd $(pwd)" >> $TFILE
  echo "mpirun -np $NP matlab -nodisplay -nojvm -r \"parImogenLoad('${RUNFILE}','${redirectStr}','${alias}', -1);\" " >> $TFILE

# This might be important, tell the user precisely how we're about to flush their cycles down the drain:
  echo "CLUSTER IMOGEN RUN SETUP:"
  echo "  Exact script that will be submitted to qsub:"
  echo "------------------------------------------------------------"
  cat $TFILE
  echo "------------------------------------------------------------"
  echo "  Into to queue $QUEUE"
  echo "  using the following invocation:"
  echo "sbatch -p $QUEUE -o $redirectStr $TFILE"
  echo -n "  In 5...";  sleep 1; echo -n "4... "; sleep 1; echo -n "3... "; sleep 1; echo -n "2... "; sleep 1; echo -n "1..."; sleep 1;

  sbatch -p $QUEUE -o $redirectStr $TFILE
  #rm $TFILE
  exit 0
fi

echo ">>> ERROR: Did not receive usable arguments. See ./imogen help:"
./imogen help

