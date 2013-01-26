% SPECIAL RUNFILE: RESUMES A PREVIOUSLY RUN MODEL

%   Run a Kojima disk model.

%--- Initialize Imogen directory ---%
starterRun();

%--- EDIT ONLY THESE TWO LINES ---%

ResumeDirectory = '~/Results/Jan13/KOJIMA_11815';
ResumeInfo.itermax = 20;
ResumeInfo.timemax = 100;
ResumeInfo.frame = 8;

%--- Setup GIS and hand Imogen the path and frame number to resume from. ---%

myIni = sprintf('%s/SimInitializer_rank%i.mat',ResumeDirectory,mpi_myrank() );
imogen(myIni, ResumeInfo);

