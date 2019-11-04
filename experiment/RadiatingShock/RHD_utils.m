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

        function N = lastValidFrame(F, x)
        % Given the dataframe F and tracked shock position x,
        % track the shock position over time.
        % compute the equilibrium shock cooling depth

        T = F.pressure(:,1,1,1) ./ F.mass(:,1,1,1);

        % logical for identifying the cold layer
        clayer = (T <= 1.05001) & (F.mass(:,1,1,1) > 1.1);

        x = x / F.dGrid{1}; % convert back to cells

        xBottomIni = find(clayer); xBottomIni = xBottomIni(1);

        hShock = xBottomIni - x(1); % height of shock in cells

        % estimate where the base of the cooling layer is
        xb = x + hShock;
    
    q = find(xb + .3*hShock > size(F.mass,1));
    

    if ~isempty(q); N=q(1); else; N = size(F.mass,4); end

        end

        function f = walkdown(x, i, itermax)
            % f = walkdown(x, i) accepts shock position vector x and initial index i
            % It walks i by one until it reaches a local maximum (or it has made itermax moves)
            
            imax = numel(x);
            
            for N = 1:itermax
                sd = 0;
                if x(i+1) > x(i)
                    sd = 1;
                end
                
                if x(i-1) > x(i)
                    if sd
                        if x(i-1) > x(i+1)
                            i = i - 1;
                        else
                            i = i + 1;
                        end
                    else
                        i = i - 1;
                    end
                else
                    i = i + 1;
                end
                
                if i == 1; break; end
                if i == imax; break; end
                
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
            rr = squeeze(rr .* (F.pressure > 1.051*F.mass));
            
            rr = sum(rr,1);
            plot(rr ./ rr(1));
        end
        
        function rr = computeRelativeLuminosity(F, radTheta)
             rr = F.mass.^(2-radTheta) .* F.pressure.^radTheta;
            rr = squeeze(rr .* (F.pressure > 1.051*F.mass));
            
            rr = sum(rr,1);
            rr = rr / rr(1);
        end
        
        function p = parseDirectoryName(x)
            if nargin < 1
                x=pwd();
            end
           
            % remove leading full path if present
            s = find(x=='/');
            if ~isempty(s)
                x=x((s(end)+1):end);
            end
            s = find(x=='_');
           
            m = sscanf(x((s(3)+1):(s(4)-1)), 'ms%e');
           
            if strcmp(x(s(5)+[1 2 3]), 'rth')
                th = sscanf(x((s(5)+1):(s(6)-1)), 'rth%i') / 100;
            else
                th = sscanf(x((s(5)+1):(s(6)-1)), 'radth%i') / 100;
            end
            g =  sscanf(x((s(6)+1):end), 'gam%i');
           
            p = struct('m', m, 'theta', th, 'gamma', g);
        end
        
        function lpp = ppLuminosity(F, theta)
            rr = RHD_utils.computeRelativeLuminosity(F, theta);
            plot(rr); drawnow;
            
            %x = input('Frame range to get lpp from: ');
            %if numel(x) == 1; x(2) = numel(rr); end
            x = [50 numel(rr)];
            
            lpp = max(rr(x(1):x(2))) - min(rr(x(1):x(2)));
        end
        
        function [mag, xbar, sigma] = gaussianPeakFit(y, bin)
            q=5;
            if bin < 6; q = bin-1; end
            xi = (-q:q)';
            yi = y((bin-q):(bin+q));
            
            try
                thefit = fit(xi, yi, 'gauss1');
            catch derp
                thefit.a1 = 0; thefit.b1 = 0; thefit.c1 = 9999; 
            end
            mag = thefit.a1;
            xbar = bin+thefit.b1;
            sigma = thefit.c1;
        end
        
        function residual = chopPeakForSearch(y, idx)
            
            ip = idx; in = idx;
            for q = idx:(numel(y)-1)
                if y(q+1) > y(q); ip = q; break; end
            end
            for q = idx:-1:2
                if y(q-1) > y(q); in = q; break; end 
            end
            
            a = y(in); b = y(ip);
            residual = y;
            residual(in:ip) = a + (b-a)*((in:ip)-in)/(ip-in);
        end
        
        function residual = subtractKnownPeaks(y, fatdot)
            % fatdot = [center, mag, stdev]
            x = (1:(numel(y)))';
            
            for n = 1:size(fatdot, 1)
                q = fatdot(n,3)^-2;
                gau = fatdot(n, 2) * exp( -(x - fatdot(n,1)).^2 *q );
                y = y - gau;
            end
            
            residual = y;
        end

        function str = assignModeName(w, M, gamma, theta)
            str = '';
            modes = {'F', '1O', '2O', '3O', '4O', '5O', '6O', '7O', '8O', '9O','10O','11O'};
            
            switch gamma
                case 167
                    ftower = [.85 2.85 5 7 9 11 13 15 17 19 21 23];
                    w = w / ((1 + 1.75/M)*(1-.038*theta));
                    m = abs(w - .256*ftower);
                    q = find(m < .1);
                    if numel(q) > 0 % numel > 1 ought to be impossible but who knows 
                        str = modes{q(1)};
                    end    
                case 140
                    ftower = [.92 2.76 5 7 9 11 13 15 17 19 21 23];
                    w = w / ((1 + 2.5/M)*(1-.04*theta));
                    m = abs(w - .185*ftower);
                    q = find(m < .1);
                    if numel(q) > 0 % numel > 1 ought to be impossible but who knows 
                        str = modes{q(1)};
                    end                    
                case 129
                    ftower = [.8 2.47 4.25 6.16 9 11 13 15 17 19 21 23];
                    w = w / ((1 + 2.84/M)*(1-.06*theta));
                    m = abs(w - .171*ftower);
                    q = find(m < .1);
                    if numel(q) > 0 % numel > 1 ought to be impossible but who knows 
                        str = modes{q(1)};
                    end                    
            end
        end
        
        
        
    end%PROTECTED
    
end%CLASS
