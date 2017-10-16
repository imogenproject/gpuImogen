function C = stokesDragCoeff(Re)

if (Re < 1)
        C = 12 ./ Re;
        return;
end
if (Re > 7.845084191866316e+02)
        C = 0.22;
        return;
end

C = 12 * Re.^-0.6;

end