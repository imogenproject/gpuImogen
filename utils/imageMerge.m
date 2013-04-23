function imageMerge(prefix, ranks, padding, set, print)

if nargin < 5; print = 0; end
if nargin < 4; fprintf('In directory containing images, run:\n  imageMerge(prefix, rank ordering, # padding, [set], print=true/false);\n'); return; end

nprocessed = 0;
for n = set

  parts = [];

  for j = 1:numel(ranks)
    fname = sprintf('%s_rank_%i_%0*i.png', prefix, ranks(j), padding, n);
    parts{j} = imread(fname)';
  end

  mergedH = [];

  for k = 1:size(ranks,2);
    mergedV = [];
    for j = 1:size(ranks,1);
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


  outname = sprintf('%s_%0*i.png', prefix, padding, n);
  imwrite(mergedH', jet(256), outname, 'png');

  nprocessed = nprocessed + 1;

  if print
    if mod(nprocessed,50) == 0; fprintf('\n'); end
    fprintf('*'); 
  end
end

end
