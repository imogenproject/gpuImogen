function imageMerge(prefix, ranks, padding, set, scalefactor, numberOffset, skipExist, skipNonexist, print)

if nargin < 9; print = 0; end
if nargin < 8; skipNonexist = 0; end
if nargin < 7; skipExist = 0; end
if nargin < 6; numberOffset = 0; end
if nargin < 5; scalefactor = 1; end
if (nargin < 4) && (nargin > 0); fprintf('In directory containing images, run:\n  imageMerge(prefix, rank ordering, # padding, [set], scalefactor, #offset, skipExist, skipNonexist, print=true/false);\n'); return; end
if nargin == 0 % No input - guide user
    prefix   = input('Prefix - location relative to current directory and image type (e.g. "mass_XY") to process: ','s');
    ranks    = input('Matrix enumerating rank ordering: ');
    padding  = input('Number of #s padding end of filenames: ');
    set      = input('Set of file numbers to process (e.g. 0:50): ');
    scalefactor  = input('Integer divisor to scale output down by (or 1 for no rescale): ');
    numberOffset = input('Number to add to output file numbers relative to input file number (e.g. offset = 10 -> mass_xy_rank_*_00.png becomes mass_xy_10.png): ');
    skipExist    = input('Skip writing new file if output file exists (0/1)? ');
    skipNonexist = input('Skip element w/o error if input does not exist? (0/1)? ');
    print    = input('1 to print *s for each image converted: ');
end

%nprocessed = 0;
parfor n = set

  parts = [];

  outname = sprintf('%s_%0*i.png', prefix, padding, n+numberOffset);

  % If the file exists and we're not overwriting, skip
  if (exist(outname,'file') ~= 0) && (skipExist == 1); continue; end

  % Try to read the inputs. If this fails, either continue silently or error
  % depending on skipNonexist
  try
  for j = 1:numel(ranks)
    fname = sprintf('%s_rank_%i_%0*i.png', prefix, ranks(j), padding, n);
    parts{j} = imread(fname)';
  end

  mergedH = [];

  for k = 1:size(ranks,2)
    mergedV = [];
    for j = 1:size(ranks,1)
      tmp = parts{ j+size(ranks,1)*(k-1)  };
      % Trim the boundary exchanges off
      if j > 1;             tmp=tmp(4:end,:); end
      if j < size(ranks,1); tmp=tmp(1:(end-3),:); end
      if k > 1;             tmp=tmp(:,4:end); end
      if k < size(ranks,2); tmp=tmp(:,1:(end-3)); end

      mergedV = [mergedV; tmp ];
    end

    mergedH = [mergedH mergedV];
  end

  % We now have the dataframe; reduce it if desired.
  if scalefactor ~= 1
    mergedH = mergedH(1:scalefactor:end, 1:scalefactor:end);
  end

  imwrite(mergedH', jet(256), outname, 'png');

 % nprocessed = nprocessed + 1;
  if print
%    if mod(nprocessed,50) == 0; fprintf('\n'); end
    fprintf('*'); 
  end

  catch READ_ERROR
    if skipNonexist == 0; rethrow READ_ERROR; end
  end

end

end
