function isUnphysical = checkPhysicality(fluids)
% isUnphysical = checkPhysicality(fluids) returns
% 1 if any cell in any fluid has negative density or energy
% 0 iff every cell in every fluid has positive density and energy

isUnphysical = 0;

for N = 1:numel(fluids)
   if any(fluids(N).mass.array(:) < 0); isUnphysical = 1; end
   if any(isnan(fluids(N).mass.array(:))); isUnphysical = 1; end
   if any(fluids(N).ener.array(:) < 0); isUnphysical = 1; end
   if any(isnan(fluids(N).ener.array(:))); isUnphysical = 1; end
end

end
