% SPECIAL RUNFILE: RESUMES A PREVIOUSLY RUN MODEL

%   Run a Kojima disk model.

%--- Initialize Imogen directory ---%
starterRun();

%--- EDIT ONLY THESE LINES ---%

ResumeDirectory = '~/Results/Apr13/KOJIMA_41815';
ResumeInfo.itermax = 10000;
ResumeInfo.timemax = 10000;
ResumeInfo.frame = 3750;

%--- Setup GIS and hand Imogen the path and frame number to resume from. ---%

myIni = sprintf('%s/SimInitializer_rank%i.mat',ResumeDirectory,mpi_myrank() );
imogen(myIni, ResumeInfo);

