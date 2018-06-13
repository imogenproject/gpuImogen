function [kx, omega, kxOffset, omegaOffset, kxRes, omegaRes] = analyzePerturbedQ(dq, x, t, linearFrames, preorpost)
% > dq: Perturbed quantity
% > x: x position values for dq(ymode#, z mode#, x value, t value)
% > t: t values for ...
% > linearFrames: The set of frames believed to be behaving linearly

yran = size(dq,1);
zran = size(dq,2);

omega = zeros([yran, zran]); omegaRes = zeros([yran, zran]);
omegaOffset = zeros([yran, zran]);
kx = zeros([yran, zran]);
kxRes = zeros([yran, zran]);
kxOffset = zeros([yran, zran]);

for u = 1:yran; for v = 1:zran
    if strcmp(preorpost,'post')
        
        [wimfit, confidenceIm] = monovariateFit(t(linearFrames), mean(squish(log(abs(dq(u,v,2:15,linearFrames))))));
        [wrefit, confidenceRe] = monovariateFit(t(linearFrames), mean(unwrap(squish(angle(dq(u,v,2:15,linearFrames))),1,2 )));

        [kximfit, confidenceKxIm] = monovariateFit(x, mean(squish(log(abs(dq(u,v,:,linearFrames)))),2));
        [kxrefit, confidenceKxRe] = monovariateFit(x, mean(unwrap(squish(angle(dq(u,v,:,linearFrames))),pi,1),2));
    else
        [wimfit, confidenceIm] = monovariateFit(t(linearFrames), mean(squish(log(abs(dq(u,v,(end-10):(end-1),linearFrames))))));
        [wrefit, confidenceRe] = monovariateFit(t(linearFrames), mean(unwrap(squish(angle(dq(u,v,(end-10):(end-1),linearFrames))),1,2 )));

        [kximfit, confidenceKxIm] = monovariateFit(x, mean(squish(log(abs(dq(u,v,:,linearFrames)))),2));
        [kxrefit, confidenceKxRe] = monovariateFit(x, mean(unwrap(squish(angle(dq(u,v,:,linearFrames))),pi,1),2));
    end
    omega(u,v) = wrefit(1) + 1i*wimfit(1);
    omegaRes(u,v) = confidenceRe.normr + 1i*confidenceIm.normr;
    omegaOffset(u,v) = wrefit(2) + 1i*wimfit(2);

    kx(u,v) = kxrefit(1) + 1i*kximfit(1);
    kxRes(u,v) = confidenceKxRe.normr + 1i*confidenceKxIm.normr;
    kxOffset(u,v) = kxrefit(2) + 1i*kximfit(2);
 
end; end

omega(isnan(real(omega))) = 1i*imag(omega(isnan(real(omega))));
omega(isnan(imag(omega))) = real(omega(isnan(imag(omega))));

kx(isnan(real(kx))) = 1i*imag(kx(isnan(real(kx))));
kx(isnan(imag(kx))) = real(kx(isnan(imag(kx))));

omegaOffset(isnan(real(omegaOffset))) = 1i*imag(omegaOffset(isnan(real(omegaOffset))));
omegaOffset(isnan(imag(omegaOffset))) = real(omegaOffset(isnan(imag(omegaOffset))));

kxOffset(isnan(real(kxOffset))) = 1i*imag(kxOffset(isnan(real(kxOffset))));
kxOffset(isnan(imag(kxOffset))) = real(kxOffset(isnan(imag(kxOffset))));


end

function [fit, residual] = monovariateFit(x, y)
%[fit, residual] = polyfit(x, y, 1);

x = x(:); y = y(:); %make it nx1

N = isnan(y);
if any(N); x = x(~N); y = y(~N); end

weight = ones(size(x));

N = numel(x);

soln = [sum(weight),  sum(x.*weight); sum(x.*weight),  sum(x.^2.*weight) ]^-1 * [sum(y.*weight); sum(x.*y.*weight) ];

fit = [soln(2) soln(1)];

residual.normr = sqrt(sum(y - (fit(2) + fit(1)*x)));

end
