% SPECIAL RUNFILE: RESUMES A PREVIOUSLY RUN MODEL

%   Run a Kojima disk model.

%--- Initialize Imogen directory ---%
starterRun();

%--- EDIT ONLY THESE LINES ---%

ResumeDirectory = '~/Results/Nov13/RADHD_B1316_RHD_ms4_ang0_minmod';
ResumeInfo.itermax = 100000;
ResumeInfo.timemax = 999999;
ResumeInfo.frame = 20000;

%--- Setup GIS and hand Imogen the path and frame number to resume from. ---%

myIni = sprintf('%s/SimInitializer_rank%i.mat',ResumeDirectory,mpi_myrank() );
imogen(myIni, ResumeInfo);

