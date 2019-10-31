function x = trackBase(mass, X)

xi = size(mass,1);

for x = xi:-1:3
    if (mass(x)-mass(x-1)) > 0
        break;
    end
end

x = X(x);

end
