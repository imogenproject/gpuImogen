function imageGen(prefix, ranks, padding, set, print)

if nargin < 5; print = 0; end

nprocessed = 0;


for n = set

  parts = [];

  for j = 1:numel(ranks)
    frame = util_LoadFrameSegment(prefix, padding, ranks(j), n);
    parts{j} = frame.mass(:,:,end/2);
    fprintf('*');
  end
  fprintf('\n');

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


  mergedH = (mergedH - min(min(mergedH)))./(max(max(mergedH)) - min(min(mergedH)));

  outname = sprintf('%s_%0*i.png', prefix, padding, n);
  imwrite(mergedH', outname, 'png');

  nprocessed = nprocessed + 1;

  if print
    if mod(nprocessed,50) == 0; fprintf('\n'); end
    fprintf('X'); 
  end
end

end


