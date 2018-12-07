function isUnphysical = checkPhysicality(fluids)
% isUnphysical = checkPhysicality(fluids) returns
% 1 if any cell in any fluid has negative density or energy
% 0 iff every cell in every fluid has positive density and energy

isUnphysical = 0;

s = size(fluids(1).mass.array);

for N = 1:numel(fluids)
   theta = (fluids(N).mass.array(:) < 0);
   if any(theta);
     SaveManager.logAllPrint('checked fluid %i for rho < 0: Problem!\n', int32(N));
     isUnphysical = 1;
     printRant(theta, s);
   end
   
   theta = isnan(fluids(N).mass.array(:));
   if any(theta)
     SaveManager.logAllPrint('checked fluid %i for rho is NaN: Problem!\n', int32(N));
     isUnphysical = 1;
     printRant(theta, s);
   end

   theta = (fluids(N).ener.array(:) < 0);
   if any(theta)
     SaveManager.logAllPrint('checked fluid %i for E < 0: Problem!\n', int32(N));
      isUnphysical = 1;
      printRant(theta, s);
   end

   theta = isnan(fluids(N).ener.array(:));
   if any(theta)
     SaveManager.logAllPrint('checked fluid %i for E is NaN: Problem!\n', int32(N));
      isUnphysical = 1;
      printRant(theta, s)
   end

end

end

function printRant(theta, dims)

b = find(theta);
x = numel(b);
SaveManager.logAllPrint('Total of %f invalid elements.\n', x);

if numel(dims)==2; dims(3)=1; end

if x > 50; b = b(1:50); end

b = b-1; % convert to zero based indices

nxy = dims(1)*dims(2);

z = floor(b/nxy);
y = floor((b-z*nxy)/dims(1));
x = b-z*nxy-dims(1)*y;

disp([b x y z]);

end
