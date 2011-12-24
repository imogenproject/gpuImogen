function vanleerLimiter(flux, dLeft, dRight)
% This function uses the Van-Leer flux limiter to average out any non-monotonic artifacts in the
% input flux value and return an appropriate Total Variation Diminishing (TVD) result.
%
%>< flux     Array of current flux values.                                  FluxArray
%>> dLeft    Differences between left fluxVals.                             double(Nx,Ny,Nz)
%>> dRight   Differences between right fluxVals.                            double(Nx,Ny,Nz)

flux.array = GPU_Type(flux.gputag) + harmonicmean(dLeft, dRight);

end
