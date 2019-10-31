function f = walkdown(x, i, itermax)
% f = walkdown(x, i) accepts vector x and initial index i

L = numel(x);

% check corner cases
if i < 1; i = 1; disp('Warning: clamped i < 1 to 1'); end
if i > L; i = L; disp('Warning: clamped i > length to length'); end

for N = 1:itermax
    if x(i+1) > x(i)
        i = i + 1;
    elseif x(i-1) > x(i)
        i = i - 1;
    end
    
    % Avoid crashing in edge cases
    if (i == L) && (x(i) > x(i-1))
        disp('Warning: Hit +end of vector');
        break;
    end

    if (i == 1) && (x(i) > x(x+1))
        disp('Warning: hit -end of vector');
    end

    % intended finish
    if (x(i+1) < x(i)) && (x(i-1) < x(i))
        break;
    end
end

f = i;

end
