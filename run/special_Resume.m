% SPECIAL RUNFILE: RESUMES A PREVIOUSLY RUN MODEL

%--- EDIT ONLY THESE LINES ---%

ResumeDirectory = '/mnt/superfast/erik-k/Results/Jun19/RADHD_6152_RHD_ms8_ang0_radth5_gam140';
ResumeInfo.itermax = 2000000;
ResumeInfo.timemax = 1e5;
ResumeInfo.frame = 1000000;

%--- Hand Imogen the path and frame number to resume from. ---%

ResumeInfo.directory = ResumeDirectory;
imogen(ResumeDirectory, ResumeInfo);

