classdef RHD_utils < handle
    % Class annotation template for creating new classes.
    %___________________________________________________________________________________________________
    
    %===================================================================================================
    properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
    end %PUBLIC
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
    end %PROTECTED
    
    %===================================================================================================
    methods %                                                                     G E T / S E T  [M]
    end%GET/SET
    
    %===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]

    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                      P R O T E C T E D    [M]
    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %                                                 S T A T I C    [M]
        function f = walkdown(x, i, itermax)
            % f = walkdown(x, i) accepts shock position vector x and initial index i
            % It walks i by one until it reaches a local maximum (or it has made itermax moves)
            
            for N = 1:itermax
                if x(i+1) > x(i)
                    i = i + 1;
                elseif x(i-1) > x(i)
                    i = i - 1;
                end
                
                if (x(i+1) < x(i)) && (x(i-1) < x(i)) % done
                    break;
                end
            end
            
            f = i;
            
        end
        
        function ftrue = projectParabolicMinimum(x, pt, mkplot, h0)
            
            if nargin < 4; h0 = 2; end
            ap = [1, 1+h0, 1+2*h0];
            am = [1+2*h0, 1+h0, 1];
            
            % NOTE: Assumes at least 8 frames between shock bounces!!!
            xleft = x(pt - am);
            xright= x(pt + ap);
            % Convert to polynomials
            xlpoly = polyfit( - am, xleft, 2);
            xrpoly = polyfit( + ap, xright,2);
            % Solve intersection point: Assume it is the small-x one
            deltapoly = xlpoly - xrpoly;
            
            alpha = roots(deltapoly);
            [~, idx] = sort(abs(alpha),'ascend');
            ftrue = pt+alpha(idx(1));
            
            if mkplot
                q = -3:.25:3;
                hold on;
                plot(pt - am, xleft, 'k*');
                plot(pt + ap, xright, 'k*');
                plot(pt + q, polyval(xlpoly, q));
                plot(pt + q, polyval(xrpoly, q));
            end
            
        end
        
        function plotRR(F, radTheta)
            % rr = rho^2 T^theta
            
            rr = F.mass.^(2-radTheta) .* F.pressure.^radTheta;
            rr = squeeze(rr .* (F.pressure > 1.05*F.mass));
            
            rr = sum(rr,1);
            plot(rr ./ rr(1));
        end
        
        function rr = computeRelativeLuminosity(F, radTheta)
             rr = F.mass.^(2-radTheta) .* F.pressure.^radTheta;
            rr = squeeze(rr .* (F.pressure > 1.05*F.mass));
            
            rr = sum(rr,1);
            rr = rr / rr(1);
        end
        
    end%PROTECTED
    
end%CLASS
