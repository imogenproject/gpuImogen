% SPECIAL RUNFILE: RESUMES A PREVIOUSLY RUN MODEL

%--- EDIT ONLY THESE LINES ---%
dirPrefix='/mnt/superfast/erik-k/Results/Jul19/done';
dirs = {'dir1', 'dir2'};
frameno = [10000, 20000];
addframes = 5000;
% Type of frame to read: Useful if e.g. a 2D sim is only saving XY data
% Note that this will almost certainly crash if the loaded frame is a subset of the actual sim
frametype = '2D_XY';
appendingRHD = 1;

%--- EDIT ONLY THESE LINES ---%

% The content of the ResumeInfo structure:
% .frame      - Mandatory, the frame # to load
% .frameType  - Mandatory, the frame type ('1D_X','1D_Y','1D_Z','2D_XY','2D_XZ','2D_YZ','3D_XYZ')
% .directory  - The absolute path to the directory that is being resumed

% Optional entries: 
% .itermax    - Changes the simulation max iteration count
% .addframes  - Changes the simulation max iteration count to (existing itermax + addframes)
%               Overwrites ResumeInfo.itermax if present
% .timemax    - Changes the simulation max time
% .scaleSaverate - changes .PERSLICE to .PERSLICE / .scaleSaverate
%               This is beyond the automatic rescaling by (old itermax)/(new itermax)
            
for P = 1:4:numel(dirs)
    ResumeDirectory = [dirPrefix '/' dirs{P}];
    
    if exist('addframes','var')
        ResumeInfo.addframes = addframes;
    else
        if numel(runto) > 1; m = runto(P); else; m = runto; end
        ResumeInfo.itermax   = m;
    end
    
    if numel(frameno) > 1; m = frameno(P); else; m = frameno; end
    ResumeInfo.frame     = m;
    ResumeInfo.frameType = frametype;
%    ResumeInfo.scaleSaverate = 5;

    %--- Hand Imogen the path and frame number to resume from. ---%
    ResumeInfo.directory = ResumeDirectory;
    outdir = imogen(ResumeDirectory, ResumeInfo);

    cd(outdir);
    if appendingRHD
        !rm savefileIndex.mat
        G = util_LoadEntireSpacetime({'mass','momX','momY','momZ','ener'}, '3D_XYZ');
        load '4D_XYZT';
        
        if ~isa(F, 'DataFrame'); F = DataFrame(F); end
        F.concatFrame(G);
        save('4D_XYZT','F');
        !rm 3D_XYZ*h5
    end

    fprintf("COMPLETED RESUMED RUN %i/%i\n", P, numel(dirs));
end

