function einfeldtFTSPlotter(q)

if isa(q,'struct')
    x = q;
else
    load q;
    x = TestResult.einfeldt;
end
q0 = 6;

figure(q0); hold on;
figure(q0+1); hold on;

for j = 1:numel(x.N)
   S = SavefilePortal(x.paths{j});
   F = S.jumpToLastFrame();
  
   xpts = ((1:x.N(j))' - (x.N(j)/2) + 0.5)*F.dGrid{1};
   
   figure(q0); plot(xpts,F.mass);
   
   figure(q0+1);
   rhot = einfeldtSolution(xpts, 1, sqrt(1.4)*5.5, 1, 1.4, sum(F.time.history));
   plot(xpts,rhot - F.mass);
end



end