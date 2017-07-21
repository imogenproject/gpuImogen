figure(1);
subplot(2,2,1);
imagesc(squeeze(log10(abs(MASS(:,1,:)-run.fluid(1).mass.array(:,1,:))))); colorbar;
subplot(2,2,2);
imagesc(squeeze(log10(abs(ENER(:,1,:)-run.fluid(1).ener.array(:,1,:))))); colorbar
subplot(2,2,3);
imagesc(squeeze(log10(abs(MOMX(:,1,:)-run.fluid(1).mom(1).array(:,1,:))))); colorbar;
subplot(2,2,4);
imagesc(squeeze(log10(abs(MOMZ(:,1,:)-run.fluid(1).mom(3).array(:,1,:))))); colorbar;
