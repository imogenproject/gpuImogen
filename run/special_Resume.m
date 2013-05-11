% SPECIAL RUNFILE: RESUMES A PREVIOUSLY RUN MODEL

%   Run a Kojima disk model.

%--- Initialize Imogen directory ---%
starterRun();

%--- EDIT ONLY THESE TWO LINES ---%

ResumeDirectory = '~/Results/May13/OTVortex_5915';
ResumeInfo.itermax = 100000;
ResumeInfo.timemax = .48;
ResumeInfo.frame = 4690;

%--- Setup GIS and hand Imogen the path and frame number to resume from. ---%

myIni = sprintf('%s/SimInitializer_rank%i.mat',ResumeDirectory,mpi_myrank() );
imogen(myIni, ResumeInfo);

