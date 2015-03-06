figure(1);
subplot(2,2,3); imagesc(mass.array - MASS); title('mass difference'); colorbar
subplot(2,2,1); imagesc(mom(1).array - MOMX); title('mom(1) difference'); colorbar
subplot(2,2,2); imagesc(mom(2).array - MOMY); title('mom(2) difference'); colorbar;
subplot(2,2,4); imagesc(ener.array - ENER); title('ener difference'); colorbar;
