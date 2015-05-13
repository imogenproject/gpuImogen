function alpha = fourierSearch(peaks, drives, order, tol)
% fourierSearch(peaks, drives, order, tol) will trawl through the vector of peaks,
% searching for linear integer sums of the elements in drives which equal
% the peaks within +-tol when the elements of the coefficient are ints from
% 0 to order

biglist =(-order(1):order(1))';

% Generate the gigantic tensor product
if numel(drives) > 1;
   for q =  2:numel(drives)
      biglist = opw(biglist, (-order:order)); 
   end
end

F = [];
for q = 1:size(biglist,1)
    F(q) = drives*biglist(q,:)';
end

[q idx] = unique(abs(F));

F = abs(F(idx)');
biglist = biglist(idx,:);

disp('Frequencies are: ');
disp(drives);

unexplained = ones(size(peaks));

for N = 1:numel(peaks);
    matches = (abs(peaks(N) - F) <= tol);
   if any(matches)
      idx = find(matches);
         for j = 1:size(idx,1)
             disp(['Suspect that [' num2str(biglist(idx(j),:)) '] -> F=' num2str(F(idx(j))) ' corresponds to F=' num2str(peaks(N))]);
         end
         unexplained(N)=0;
   end
end

disp('No apparent explanation for these:');
disp(peaks((unexplained==1)));

end

function yay = opw(things, new)

q = size(things, 1);

yay = zeros([q*numel(new) (size(things,2)+1)]);

for n = 1:size(things,2);
   [u v] =  ndgrid(things(:,n), new);
   yay(:,n) = u(:);
   if n == 1; yay(:,end) = v(:); end
end

end