function [drho dV dB deps] = evaluateEigenvector(EV, KdotX, hphi )

    dV = zeros([3 size(KdotX,1) size(KdotX,2) size(KdotX,3)]);
    dB = zeros([3 size(KdotX,1) size(KdotX,2) size(KdotX,3)]);

    drho        = real(EV(1)*exp(1i*KdotX));
    dV(1,:,:,:) = real(EV(2)*exp(1i*KdotX));
    dV(2,:,:,:) = real(EV(3)*exp(1i*KdotX));
    dV(3,:,:,:) = real(EV(4)*exp(1i*KdotX));
    % We have to be a mite careful here since these are not cell-centered

% HACK: just circshift the K.X if we're assuming it's a square
    dB(1,:,:,:) = real(EV(5)*exp(1i*KdotX-hphi(1)));
    dB(2,:,:,:) = real(EV(6)*exp(1i*KdotX-hphi(2)));
    dB(3,:,:,:) = real(EV(7)*exp(1i*KdotX-hphi(3)));
    deps        = real(EV(8)*exp(1i*KdotX));

end
